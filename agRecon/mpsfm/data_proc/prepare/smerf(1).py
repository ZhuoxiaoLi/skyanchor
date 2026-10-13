import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from mpsfm.data_proc.smerf import SMERFDataset


def main(delete_files: bool = True):
    """
    Main function to:
    1) Download SMERF archives if needed.
    2) Extract and process each scene.
    3) Scale cameras and points2D.
    4) Write out updated pycolmap Reconstructions.
    If delete_files is True, temporary files and archives are removed.
    """
    data_dir = SMERFDataset.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
        "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip",
    ]

    # Download each archive
    for url in urls:
        print(f"Downloading {Path(url).stem}...")
        subprocess.run(["wget", "-P", str(data_dir), url], check=True)

    # Process each scene
    for scene in SMERFDataset.scenes:
        print(f"Processing {scene}...")
        zip_file = data_dir / f"{scene}.zip"

        # Unzip selected folders into data_dir
        subprocess.run(
            ["unzip", str(zip_file), f"{scene}/images_2/*", f"{scene}/sparse/*", "-d", str(data_dir)], check=True
        )

        # Rename "images_2" folder to "images"
        (data_dir / scene / "images_2").rename(data_dir / scene / "images")

        # Rename "sparse" folder for reconstruction
        recdir = data_dir / scene / "sparse"
        tmp_recdir = recdir.parent / "sparse_tmp"
        recdir.rename(tmp_recdir)

        # Load reconstruction
        rec = pycolmap.Reconstruction(tmp_recdir / "0")

        # Scale camera parameters
        print("\tScaling camera params...")
        for cam in rec.cameras.values():
            w, h = cam.width, cam.height
            cam.width, cam.height = w // 2, h // 2
            sx, sy = cam.width / w, cam.height / h
            cam.params = [p * s for p, s in zip(cam.params, 2 * [sx, sy])]

        # Scale points2D
        print("\tScaling points2D...")
        for im in tqdm(rec.images.values()):
            for p2d in im.points2D:
                p2d.xy *= np.array([sx, sy])

        # Save the updated reconstruction
        recdir = recdir.parent / "rec"
        recdir.mkdir(parents=True, exist_ok=True)
        rec.write(recdir)

        # Remove temporary reconstruction directory
        if delete_files:
            shutil.rmtree(tmp_recdir)
            zip_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMERF dataset preparation script.")
    parser.add_argument(
        "--delete-files",
        action="store_true",
        default=False,
        help="If set, temporary directories and archives are removed after processing.",
    )
    args = parser.parse_args()

    main(delete_files=args.delete_files)
