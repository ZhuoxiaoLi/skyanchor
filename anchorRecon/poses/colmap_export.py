from __future__ import annotations

import os

import numpy as np
import torch

from dataloaders.read_write_model import BaseImage, Camera, Point3D, rotmat2qvec, write_model


def _as_numpy(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _empty_xys():
    return np.zeros((0, 2), dtype=np.float64)


def _empty_ids():
    return np.zeros((0,), dtype=np.int64)


def build_final_ba_colmap_model(
    keyframes,
    sparse_payloads,
    pose_info_key: str | None = None,
    focal_info_key: str | None = None,
):
    ordered_keyframes = sorted(list(keyframes), key=lambda keyframe: keyframe.index)
    image_id_by_obj = {}
    image_id_by_index = {}
    cameras = {}
    images = {}
    image_observations = {}

    for image_id, keyframe in enumerate(ordered_keyframes, start=1):
        if focal_info_key is None:
            focal_value = keyframe.f
        else:
            focal_value = keyframe.info.get(focal_info_key)
            if focal_value is None:
                raise ValueError(
                    f"Keyframe {keyframe.info.get('name', keyframe.index)} has no "
                    f"'{focal_info_key}' focal snapshot."
                )
        focal = float(torch.as_tensor(focal_value).reshape(-1)[0].item())
        centre = _as_numpy(keyframe.centre, np.float64)
        cameras[image_id] = Camera(
            id=image_id,
            model="PINHOLE",
            width=int(keyframe.width),
            height=int(keyframe.height),
            params=np.array([focal, focal, centre[0], centre[1]], dtype=np.float64),
        )
        image_observations[image_id] = []
        image_id_by_obj[id(keyframe)] = image_id
        image_id_by_index[keyframe.index] = image_id

    points3D = {}
    next_point3d_id = 1
    for payload in sparse_payloads:
        if not payload:
            continue

        payload_keyframes = payload.get("keyframes", [])
        payload_uvs = _as_numpy(payload.get("uvs"), np.float64)
        payload_points = _as_numpy(payload.get("points3d"), np.float64)
        payload_rgb = _as_numpy(payload.get("rgb"), np.uint8)
        if (
            payload_uvs is None
            or payload_points is None
            or payload_uvs.ndim != 3
            or payload_points.ndim != 2
            or payload_uvs.shape[0] != payload_points.shape[0]
        ):
            continue
        if payload_rgb is None or len(payload_rgb) != len(payload_points):
            payload_rgb = np.zeros((len(payload_points), 3), dtype=np.uint8)

        payload_image_ids = []
        for keyframe in payload_keyframes:
            image_id = image_id_by_obj.get(id(keyframe), image_id_by_index.get(keyframe.index))
            payload_image_ids.append(image_id)

        for local_point_idx, xyz in enumerate(payload_points):
            pending_observations = []
            for local_cam_idx, image_id in enumerate(payload_image_ids):
                if image_id is None:
                    continue
                uv = payload_uvs[local_point_idx, local_cam_idx]
                if not np.isfinite(uv).all() or np.any(uv < 0):
                    continue
                pending_observations.append((image_id, uv))

            if len(pending_observations) < 2:
                continue

            image_ids = []
            point2D_idxs = []
            for image_id, uv in pending_observations:
                point2D_idx = len(image_observations[image_id])
                image_observations[image_id].append((uv, next_point3d_id))
                image_ids.append(image_id)
                point2D_idxs.append(point2D_idx)

            points3D[next_point3d_id] = Point3D(
                id=next_point3d_id,
                xyz=xyz.astype(np.float64, copy=False),
                rgb=np.clip(payload_rgb[local_point_idx], 0, 255).astype(np.uint8, copy=False),
                error=0.0,
                image_ids=np.asarray(image_ids, dtype=np.int32),
                point2D_idxs=np.asarray(point2D_idxs, dtype=np.int32),
            )
            next_point3d_id += 1

    for image_id, keyframe in enumerate(ordered_keyframes, start=1):
        if pose_info_key is None:
            Rt_value = keyframe.get_Rt()
        else:
            Rt_value = keyframe.info.get(pose_info_key)
            if Rt_value is None:
                raise ValueError(
                    f"Keyframe {keyframe.info.get('name', keyframe.index)} has no "
                    f"'{pose_info_key}' pose snapshot."
                )
        Rt = _as_numpy(Rt_value, np.float64)
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        observations = image_observations[image_id]
        if observations:
            xys = np.asarray([uv for uv, _ in observations], dtype=np.float64)
            point3D_ids = np.asarray([point3d_id for _, point3d_id in observations], dtype=np.int64)
        else:
            xys = _empty_xys()
            point3D_ids = _empty_ids()
        images[image_id] = BaseImage(
            id=image_id,
            qvec=-rotmat2qvec(R),
            tvec=t,
            camera_id=image_id,
            name=keyframe.info.get("name", f"{image_id:06d}.png"),
            xys=xys,
            point3D_ids=point3D_ids,
        )

    return cameras, images, points3D


def export_final_ba_colmap_text(
    export_dir: str,
    keyframes,
    sparse_payloads,
):
    os.makedirs(export_dir, exist_ok=True)
    cameras, images, points3D = build_final_ba_colmap_model(
        keyframes=keyframes,
        sparse_payloads=sparse_payloads,
    )
    write_model(cameras, images, points3D, export_dir, ext=".txt")
    n_observations = sum(len(image.point3D_ids) for image in images.values())
    return {
        "path": export_dir,
        "num_cameras": len(cameras),
        "num_images": len(images),
        "num_points3D": len(points3D),
        "num_observations": int(n_observations),
    }


def export_pose_snapshots_colmap_text(
    export_dir: str,
    keyframes,
    pose_info_key: str = "online_Rt",
    focal_info_key: str = "online_focal",
):
    """Export per-keyframe pose snapshots as a points-free COLMAP text model."""
    os.makedirs(export_dir, exist_ok=True)
    cameras, images, points3D = build_final_ba_colmap_model(
        keyframes=keyframes,
        sparse_payloads=[],
        pose_info_key=pose_info_key,
        focal_info_key=focal_info_key,
    )
    write_model(cameras, images, points3D, export_dir, ext=".txt")
    return {
        "path": export_dir,
        "num_cameras": len(cameras),
        "num_images": len(images),
        "num_points3D": 0,
        "num_observations": 0,
    }
