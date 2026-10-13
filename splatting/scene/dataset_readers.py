#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional, List
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import io
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    colmap_id: int  # COLMAP image_id, used for matching with pair.txt
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx: Optional[float] = None
    cy: Optional[float] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    if len(cam_info) == 0:
        raise ValueError(
            'No supported cameras or matching image files were found (zero views). '
            'Check that COLMAP image names under --source/-s match files in the --images directory by filename stem and case, and that all images have been copied.'
        )

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".jpe", ".heic"}

def _build_image_stem_index(images_folder: str) -> dict:
    """Index images by lowercase filename stem, ignoring extension and case."""
    from collections import defaultdict
    index = defaultdict(list)
    if not os.path.isdir(images_folder):
        return index
    for root, _dirs, files in os.walk(images_folder):
        for f in files:
            if not _is_image_file(f):
                continue
            st = os.path.splitext(f)[0].lower()
            index[st].append(os.path.join(root, f))
    return index

def _pick_from_stem_candidates(candidates: List[str], images_folder: str, extr_name: str) -> Optional[str]:
    """Prefer the candidate whose relative directory matches the COLMAP image name."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    sub = os.path.dirname(extr_name.replace("\\", "/"))
    for p in candidates:
        rel = os.path.relpath(p, images_folder)
        d = os.path.dirname(rel).replace("\\", "/")
        if d == sub:
            return p
    return min(candidates, key=len)

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    skipped_images = []
    images_folder = os.path.normpath(images_folder)
    stem_index = _build_image_stem_index(images_folder)

    for idx, key in enumerate(tqdm(cam_extrinsics, desc="Reading cameras", unit="img")):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        colmap_id = key  # COLMAP image_id, used for matching with pair.txt
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name.replace("\\", os.sep))
        image_name = os.path.splitext(os.path.basename(extr.name))[0]
        stem_key = image_name.lower()

        if not os.path.exists(image_path):
            subdir = os.path.dirname(extr.name.replace("\\", "/"))
            search_dir = os.path.join(images_folder, subdir) if subdir else images_folder
            found = None
            if os.path.isdir(search_dir):
                for f in os.listdir(search_dir):
                    if not _is_image_file(f):
                        continue
                    if os.path.splitext(f)[0].lower() == stem_key:
                        found = os.path.join(search_dir, f)
                        break
            if found is None and stem_key in stem_index:
                found = _pick_from_stem_candidates(stem_index[stem_key], images_folder, extr.name)
            if found is not None:
                image_path = found
            else:
                skipped_images.append(extr.name)
                continue

        cam_info = CameraInfo(uid=uid, colmap_id=colmap_id, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              cx=float(intr.params[-2]), cy=float(intr.params[-1]))
        cam_infos.append(cam_info)

    if skipped_images:
        print(f"[WARNING] Skipped {len(skipped_images)} missing image files:")
        for img_name in skipped_images[:10]:
            print(f"  - {img_name}")
        if len(skipped_images) > 10:
            print(f"  ... and {len(skipped_images) - 10} more files")

    return cam_infos

def _sanitize_ply_header(path):
    """Strip non-ASCII bytes from PLY header (e.g. Chinese locale timestamps from CloudCompare)."""
    with open(path, 'rb') as f:
        raw = f.read()
    end_marker = b'end_header\n'
    idx = raw.find(end_marker)
    if idx == -1:
        return io.BytesIO(raw)
    header_end = idx + len(end_marker)
    header = raw[:header_end]
    clean_header = bytes(b if b < 128 else ord('_') for b in header)
    return io.BytesIO(clean_header + raw[header_end:])

def fetchPly(path):
    plydata = PlyData.read(_sanitize_ply_header(path))
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    prop_names = {p.name for p in vertices.properties}

    if 'red' in prop_names and 'green' in prop_names and 'blue' in prop_names:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
        if colors.dtype.kind == 'f':
            colors = colors if colors.max() <= 1.0 else colors / 255.0
        else:
            colors = colors.astype(np.float64) / 255.0
    else:
        print(f"[fetchPly] WARNING: PLY missing color properties (found: {prop_names}), using random colors")
        colors = np.random.rand(positions.shape[0], 3)

    if 'nx' in prop_names and 'ny' in prop_names and 'nz' in prop_names:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        print(f"[fetchPly] WARNING: PLY missing normal properties, using zeros")
        normals = np.zeros_like(positions)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def resolve_sparse_ply_path(source_path, ply_name):
    ply_name = ply_name or "dense_v2.ply"
    if os.path.isabs(ply_name):
        return ply_name
    if os.path.dirname(ply_name):
        return os.path.abspath(os.path.join(source_path, ply_name))
    return os.path.join(source_path, "sparse", "0", ply_name)

def readColmapSceneInfo(path, images, eval, init_ply_path="", dense_init_ply_name="dense_v2.ply", llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    images_folder = os.path.join(path, reading_dir)
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(
            f"Image directory not found (set --images or place images in <source>/{reading_dir}): {os.path.abspath(images_folder)}"
        )
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_folder)
    if len(cam_infos_unsorted) == 0:
        raise ValueError(
            f"No images matching COLMAP records were found in: {os.path.abspath(images_folder)}. "
            'Filename-stem matching across common image formats was attempted. Check the image path or adjust --images.'
        )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    cams_for_norm = train_cam_infos if len(train_cam_infos) > 0 else cam_infos
    nerf_normalization = getNerfppNorm(cams_for_norm)
    dense_ply_path = resolve_sparse_ply_path(path, dense_init_ply_name)
    sfm_ply_path = os.path.join(path, "sparse/0/points3D.ply")
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if init_ply_path:
        ply_path = init_ply_path
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Custom init ply not found: {ply_path}")
        print(f"[Init] Using custom init point cloud: {ply_path}")
    elif os.path.exists(dense_ply_path):
        ply_path = dense_ply_path
        print(f"[Init] Using existing dense point cloud: {dense_ply_path}")
    else:
        print(f"[Init] Requested dense point cloud not found: {dense_ply_path}")
        ply_path = sfm_ply_path
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        print(f"[Init] Using SfM point cloud: {ply_path}, dense init will run later.")
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"[ERROR] Failed to read PLY file: {ply_path}")
        print(f"[ERROR] Exception: {e}")
        raise

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info





def load_ground_cameras(ground_path, images_subdir, resolution_scale, args, llffhold=8):
    """Load ground cameras without creating a Scene. Apply the aerial llffhold train/test split when args.eval is enabled. Return (train_cameras, test_cameras, cameras_extent)."""
    from utils.camera_utils import cameraList_from_camInfos

    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(ground_path, "sparse/0/images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(ground_path, "sparse/0/cameras.bin"))
    except:
        cam_extrinsics = read_extrinsics_text(os.path.join(ground_path, "sparse/0/images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(ground_path, "sparse/0/cameras.txt"))

    cam_infos = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(ground_path, images_subdir),
    )
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    do_eval = getattr(args, 'eval', False)
    if do_eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos  = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        print(f"[Ground] eval mode: {len(train_cam_infos)} train / {len(test_cam_infos)} test (llffhold={llffhold})")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    cameras_extent = nerf_normalization["radius"]

    train_camera_list = cameraList_from_camInfos(train_cam_infos, resolution_scale, args)
    test_camera_list = cameraList_from_camInfos(test_cam_infos, resolution_scale, args)

    return train_camera_list, test_camera_list, cameras_extent


def load_ground_point_cloud(ground_path, args):
    """Load the initialization point cloud for the ground stage."""
    dense_ply_name = (
        getattr(args, "ground_dense_init_ply_name", "")
        or getattr(args, "dense_init_ply_name", "dense_v2.ply")
    )
    dense_ply_path = resolve_sparse_ply_path(ground_path, dense_ply_name)
    sfm_ply_path = os.path.join(ground_path, "sparse/0/points3D.ply")
    if os.path.exists(dense_ply_path):
        pcd = fetchPly(dense_ply_path)
        print(f"[Ground] Loaded dense point cloud: {dense_ply_path} ({len(pcd.points)} points)")
    elif os.path.exists(sfm_ply_path):
        pcd = fetchPly(sfm_ply_path)
        print(f"[Ground] Loaded SfM point cloud: {sfm_ply_path} ({len(pcd.points)} points)")
    else:
        raise FileNotFoundError(f"[Ground] No point cloud found in {ground_path}/sparse/0/")

    return pcd


def load_ground_data(ground_path, images_subdir, resolution_scale, args, llffhold=8):
    """Load ground cameras and points without creating a Scene. Apply the llffhold evaluation split and return (train_cameras, test_cameras, point_cloud, cameras_extent)."""
    train_camera_list, test_camera_list, cameras_extent = load_ground_cameras(
        ground_path, images_subdir, resolution_scale, args, llffhold=llffhold
    )
    pcd = load_ground_point_cloud(ground_path, args)
    return train_camera_list, test_camera_list, pcd, cameras_extent
