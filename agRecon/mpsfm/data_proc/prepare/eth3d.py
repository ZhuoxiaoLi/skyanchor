import argparse
import copy
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pycolmap
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from mpsfm.data_proc.eth3d import ETH3DDataset


def get_rot90(pose_camfromworld):
    """
    Determines how many 90-degree rotations are needed based on the gravity vector in camera coordinates.
    """
    gravity_world = np.array([0, 0, -1])
    gravity_cam = pose_camfromworld.rotation.matrix() @ gravity_world
    angle = np.rad2deg(np.arctan2(gravity_cam[1], gravity_cam[0]))
    binned = np.round(angle / 90) % 4
    num_rot90 = int((binned - 1) % 4)
    return num_rot90


def rotate_image(image, rot_k):
    """
    Rotates the given image (numpy array) by 90 * rot_k degrees (k times).
    """
    return np.rot90(image, k=(rot_k) % 4, axes=(0, 1))


def rotate_cam(cfw, rot_k):
    """
    Applies k 90-degree rotations to the camera.
    """
    R_cam = cfw.matrix()
    R_cam_homogeneous = np.eye(4)
    R_cam_homogeneous[:3, :4] = R_cam
    R_90_homogeneous = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    for _ in range(rot_k):
        R_cam_homogeneous = np.linalg.inv(R_90_homogeneous) @ R_cam_homogeneous
    return R_cam_homogeneous[:3, :4]


def main(delete_files: bool = True):
    """
    Main function to:
    1) Download ETH3D archives if needed.
    2) Extract and process each scene.
    3) Resize and rotate images.
    4) Write out updated pycolmap Reconstructions.
    If delete_files is True, old directories and archives are removed.
    """
    data_dir = ETH3DDataset.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # URLs for the ETH3D dataset (training and test DSLR undistorted)
    urls = [
        "https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z",
        "https://www.eth3d.net/data/multi_view_test_dslr_undistorted.7z",
    ]

    # Download each archive to data_dir
    for url in urls:
        print(f"Downloading {Path(url).stem}...")
        subprocess.run(["wget", "-P", str(data_dir), url], check=True)

    # Extract each archive and remove the file afterwards
    for url in urls:
        archive_path = data_dir / Path(url).name
        print(f"Extracting {archive_path.name}...")
        subprocess.run(["7z", "x", str(archive_path), f"-o{data_dir}"], check=True)
        if delete_files:
            archive_path.unlink()

    # Iterate over each subdirectory in data_dir, which corresponds to a scene
    for scene_dir in data_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        print(scene_dir.name)
        images_dir = scene_dir / "images"
        images_tmp_dir = scene_dir / "images_tmp"

        # Move existing images folder to a temporary location
        images_dir.rename(images_tmp_dir)

        # Some downloads have an extra subdirectory "dslr_images_undistorted"; flatten it if found
        inner_dir = images_tmp_dir / "dslr_images_undistorted"
        if inner_dir.exists():
            for f in inner_dir.iterdir():
                f.rename(images_tmp_dir / f.name)
            inner_dir.rmdir()

        # Prepare to build a new reconstruction
        scene = scene_dir.name
        rec = pycolmap.Reconstruction(scene_dir / "dslr_calibration_undistorted")
        new_rec = pycolmap.Reconstruction()

        rgb_dir = scene_dir / "images_tmp"
        new_rgb_dir = scene_dir / "images"
        new_rgb_dir.mkdir(exist_ok=True, parents=True)
        print(f"Writing images to {new_rgb_dir}")

        # Loop over each image in the reconstruction
        for img_id, image in tqdm(rec.images.items()):
            # Decide how many 90-degree rotations to apply based on the camera orientation
            rot = get_rot90(image.cam_from_world)
            # Force no rotation for the 'electrof' scene
            if scene == "electrof":
                rot = 0

            # Copy the original image object, then load its RGB data
            new_image = copy.copy(image)
            rgb = np.array(Image.open(rgb_dir / Path(new_image.name).name).convert("RGB"))

            # Change extension to .png for the new image
            new_image.name = Path(new_image.name).with_suffix(".png").name
            H, W, _ = rgb.shape

            # Downsample image to 1/4 of its original dimensions
            rgb = to_tensor(rgb).unsqueeze(0)
            rgb = F.interpolate(rgb, size=(H // 4, W // 4), mode="area")
            rgb = (rgb.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)

            # Update height, width for the new downsampled image
            H, W, _ = rgb.shape
            # Rotate the image data if needed
            rgb_rotated = rotate_image(rgb, rot)

            # Update the camera-from-world pose to reflect the rotation
            cfw_rotated = pycolmap.Rigid3d(rotate_cam(image.cam_from_world, rot))
            new_image.cam_from_world = cfw_rotated

            # Copy the camera and adjust for new dimensions
            camera = rec.cameras[image.camera_id]
            new_camera = copy.copy(camera)
            new_camera.width, new_camera.height = W, H
            sx, sy = new_camera.width / camera.width, new_camera.height / camera.height

            # Scale the parameters to match the new resolution
            params = [p * s for p, s in zip(camera.params, 2 * [sx, sy])]
            new_camera.params = params
            new_image.reset_camera_ptr()

            # If we rotate by 90 or 270 degrees, swap camera height/width
            if rot == 1 or rot == 3:
                h, w = new_camera.height, new_camera.width
                new_camera.height, new_camera.width = w, h
                # Re-index parameter array for rotated camera
                params = [new_camera.params[i] for i in [1, 0, 3, 2]]
                new_camera.params = params

            # Assign new IDs so each image can have a separate camera if needed
            new_camera.camera_id = img_id
            new_image.camera_id = img_id

            # Adjust 2D keypoints to match rotation and new scale
            for p2d in new_image.points2D:
                p2d.xy *= np.array([sx, sy])  # scale
                h, w = H, W
                for _ in range(rot):  # rotate
                    x, y = p2d.xy
                    p2d.xy = np.array([y, w - 1 - x])
                    h, w = w, h

            # Add the new camera and the new image to the new reconstruction
            new_rec.add_camera(new_camera)
            new_rec.add_image(new_image)

            # Save the rotated (and downsampled) image data
            Image.fromarray(rgb_rotated).save(new_rgb_dir / new_image.name, pnginfo=None, optimize=False)

        # Write the final reconstruction to a new directory
        new_rec_dir = scene_dir / "rec"
        new_rec_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing new rec to {new_rec_dir}")
        new_rec.write_binary(new_rec_dir)

        # If delete_files is True, remove old directories
        if delete_files:
            shutil.rmtree(scene_dir / "dslr_calibration_undistorted")
            shutil.rmtree(scene_dir / "images_tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETH3D dataset preparation script.")
    parser.add_argument(
        "--delete-files", action="store_true", default=False, help="If set, old folders are removed after processing."
    )
    args = parser.parse_args()
    main(delete_files=args.delete_files)
