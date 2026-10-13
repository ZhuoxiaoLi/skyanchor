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

import torch
import numpy as np
from scene import Scene
from scene.cameras import Camera, MiniCam
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.colmap_loader import (
    read_extrinsics_text, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary,
)
from scene.dataset_readers import readColmapCameras
from scipy.spatial.transform import Rotation as SciRotation, Slerp
from scipy.interpolate import CubicSpline
import cv2


def get_camera_alpha_mask(view, device=None, dtype=None):
    alpha_mask = getattr(view, "alpha_mask", None)
    if alpha_mask is None:
        alpha_mask = getattr(view, "gt_alpha_mask", None)
    if alpha_mask is None:
        height = getattr(view, "image_height")
        width = getattr(view, "image_width")
        target_device = (
            device if isinstance(device, torch.device)
            else torch.device(device) if device is not None
            else torch.device("cuda")
        )
        target_dtype = dtype if dtype is not None else torch.float32
        return torch.ones((1, height, width), device=target_device, dtype=target_dtype)

    if alpha_mask.dim() == 2:
        alpha_mask = alpha_mask.unsqueeze(0)
    if device is not None:
        alpha_mask = alpha_mask.to(device)
    if dtype is not None:
        alpha_mask = alpha_mask.to(dtype=dtype)
    return alpha_mask.clamp(0.0, 1.0)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], 0.0, 1.0)
        gt = torch.clamp(view.original_image[0:3, :, :].to(rendering.device), 0.0, 1.0)
        alpha_mask = get_camera_alpha_mask(view, device=rendering.device, dtype=rendering.dtype)
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def load_colmap_cameras_sorted(colmap_path, images_dir):
    """Read COLMAP cameras and return CameraInfo entries sorted by image name."""
    try:
        cam_ext = read_extrinsics_binary(os.path.join(colmap_path, "sparse/0/images.bin"))
        cam_int = read_intrinsics_binary(os.path.join(colmap_path, "sparse/0/cameras.bin"))
    except:
        cam_ext = read_extrinsics_text(os.path.join(colmap_path, "sparse/0/images.txt"))
        cam_int = read_intrinsics_text(os.path.join(colmap_path, "sparse/0/cameras.txt"))
    cam_infos = readColmapCameras(cam_ext, cam_int, images_dir)
    return sorted(cam_infos, key=lambda c: c.image_name)


def _smooth_rotations_window(quats, window_size):
    """Smooth a quaternion sequence with a moving average."""
    n = len(quats)
    if window_size < 2 or n < 3:
        return quats
    half = window_size // 2
    smoothed = np.copy(quats)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = quats[lo:hi]
        ref = chunk[0]
        aligned = np.where((chunk * ref).sum(axis=1, keepdims=True) < 0, -chunk, chunk)
        avg = aligned.mean(axis=0)
        avg /= np.linalg.norm(avg)
        smoothed[i] = avg
    return smoothed


def build_smooth_trajectory(cam_infos, num_frames, loop=False, rot_smooth_window=5):
    """Interpolate an ordered camera trajectory using cubic splines and Slerp. Smooth keyframe rotations with an odd-sized rot_smooth_window. Return NumPy positions (N, 3) and rotations (N, 3, 3)."""
    n = len(cam_infos)
    positions = np.zeros((n, 3))
    rot_mats = np.zeros((n, 3, 3))

    for i, ci in enumerate(cam_infos):
        W2C = np.eye(4)
        W2C[:3, :3] = ci.R.T
        W2C[:3, 3] = ci.T
        C2W = np.linalg.inv(W2C)
        positions[i] = C2W[:3, 3]
        rot_mats[i] = C2W[:3, :3]

    if rot_smooth_window > 1:
        quats = SciRotation.from_matrix(rot_mats).as_quat()  # (n, 4)
        quats = _smooth_rotations_window(quats, rot_smooth_window)
        rot_mats = SciRotation.from_quat(quats).as_matrix()

    t_key = np.linspace(0, 1, n)
    t_query = np.linspace(0, 1, num_frames)

    if loop:
        positions = np.vstack([positions, positions[0:1]])
        rot_mats = np.vstack([rot_mats, rot_mats[0:1]])
        t_key = np.linspace(0, 1, n + 1)

    cs = CubicSpline(t_key, positions, bc_type='clamped' if not loop else 'periodic')
    smooth_pos = cs(t_query)

    sci_rots = SciRotation.from_matrix(rot_mats)
    slerp = Slerp(t_key, sci_rots)
    smooth_rots = slerp(t_query).as_matrix()

    return smooth_pos, smooth_rots


def render_video(
    model_path, iteration, gaussians, pipeline, background,
    cam_infos, duration, fps, output_name="trajectory_video", loop=False,
    rot_smooth_window=5,
):
    """Render a video along a smooth camera trajectory."""
    num_frames = int(duration * fps)
    ref = cam_infos[0]

    smooth_pos, smooth_rots = build_smooth_trajectory(
        cam_infos, num_frames, loop=loop, rot_smooth_window=rot_smooth_window)

    video_dir = os.path.join(model_path, "videos")
    frames_dir = os.path.join(video_dir, f"{output_name}_frames")
    makedirs(frames_dir, exist_ok=True)

    video_path = os.path.join(video_dir, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (ref.width, ref.height))

    print(f"[Video] Rendering {num_frames} frames ({duration}s @ {fps}fps), "
          f"resolution {ref.width}x{ref.height}...")
    for i in tqdm(range(num_frames), desc="Rendering video"):
        C2W = np.eye(4, dtype=np.float32)
        C2W[:3, :3] = smooth_rots[i]
        C2W[:3, 3] = smooth_pos[i]
        W2C = np.linalg.inv(C2W)

        R_cam = W2C[:3, :3].T
        T_cam = W2C[:3, 3]

        w2v = torch.tensor(getWorld2View2(R_cam, T_cam)).transpose(0, 1).cuda()
        proj = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=ref.FovX, fovY=ref.FovY
        ).transpose(0, 1).cuda()
        full_proj = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

        cam = MiniCam(
            width=ref.width, height=ref.height,
            fovy=ref.FovY, fovx=ref.FovX,
            znear=0.01, zfar=100.0,
            world_view_transform=w2v,
            full_proj_transform=full_proj,
        )

        rendering = render(cam, gaussians, pipeline, background)["render"]

        torchvision.utils.save_image(
            rendering, os.path.join(frames_dir, f"{i:05d}.png"))

        frame_np = (torch.clamp(rendering, 0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[Video] Saved: {video_path}")
    print(f"[Video] Frames also saved at: {frames_dir}")
    return video_path


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        aerial_train = list(scene.getTrainCameras())
        aerial_test = list(scene.getTestCameras())
        ground_train, ground_test = [], []

        ground_source = getattr(dataset, 'ground_source_path', '')
        if ground_source and ground_source.strip() and os.path.exists(ground_source):
            from scene.dataset_readers import load_ground_data
            ground_images_sub = getattr(dataset, 'ground_images', 'images')
            ground_train, ground_test, _, _ = load_ground_data(
                ground_source, ground_images_sub,
                resolution_scale=1.0, args=dataset,
            )
            print(f"[Render] Loaded ground: {len(ground_train)} train + {len(ground_test)} test")

        has_ground = bool(ground_train) or bool(ground_test)

        if not skip_train:
            if aerial_train:
                name = "train_aerial" if has_ground else "train"
                render_set(dataset.model_path, name, scene.loaded_iter,
                           aerial_train, gaussians, pipeline, background)
            if ground_train:
                render_set(dataset.model_path, "train_ground", scene.loaded_iter,
                           ground_train, gaussians, pipeline, background)

        if not skip_test:
            if aerial_test:
                name = "test_aerial" if has_ground else "test"
                render_set(dataset.model_path, name, scene.loaded_iter,
                           aerial_test, gaussians, pipeline, background)
            if ground_test:
                render_set(dataset.model_path, "test_ground", scene.loaded_iter,
                           ground_test, gaussians, pipeline, background)
            if not (aerial_test or ground_test):
                print("[Render] No test cameras (did you train with --eval ?)")


def render_trajectory_video(
    dataset: ModelParams, iteration: int, pipeline: PipelineParams,
    trajectory_source: str, duration: float, fps: int,
    trajectory_images: str, output_name: str, loop: bool,
    rot_smooth_window: int = 5,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        images_dir = os.path.join(trajectory_source, trajectory_images)
        cam_infos = load_colmap_cameras_sorted(trajectory_source, images_dir)
        print(f"[Video] Loaded {len(cam_infos)} cameras from {trajectory_source}")
        print(f"[Video] Camera order: {cam_infos[0].image_name} → {cam_infos[-1].image_name}")
        print(f"[Video] Rotation smooth window: {rot_smooth_window}")

        render_video(
            model_path=dataset.model_path,
            iteration=scene.loaded_iter,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
            cam_infos=cam_infos,
            duration=duration,
            fps=fps,
            output_name=output_name,
            loop=loop,
            rot_smooth_window=rot_smooth_window,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--render_video", action="store_true",
                        help='Render a camera-trajectory video')
    parser.add_argument("--trajectory_source", type=str, default="",
                        help='Trajectory source directory containing a COLMAP sparse/0 model')
    parser.add_argument("--trajectory_images", type=str, default="images",
                        help='Image subdirectory for the trajectory source')
    parser.add_argument("--video_duration", type=float, default=10.0,
                        help='Video duration in seconds')
    parser.add_argument("--video_fps", type=int, default=30,
                        help='Video frame rate')
    parser.add_argument("--video_name", type=str, default="trajectory_video",
                        help='Output video filename')
    parser.add_argument("--video_loop", action="store_true",
                        help='Connect the trajectory endpoints to form a loop')
    parser.add_argument("--rot_smooth_window", type=int, default=5,
                        help='Rotation smoothing window (odd; 1 disables smoothing)')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    if getattr(args, 'render_video', False) and getattr(args, 'trajectory_source', ''):
        render_trajectory_video(
            dataset=model.extract(args),
            iteration=args.iteration,
            pipeline=pipeline.extract(args),
            trajectory_source=os.path.abspath(args.trajectory_source),
            duration=args.video_duration,
            fps=args.video_fps,
            trajectory_images=args.trajectory_images,
            output_name=args.video_name,
            loop=args.video_loop,
            rot_smooth_window=args.rot_smooth_window,
        )
    else:
        render_sets(
            model.extract(args), args.iteration,
            pipeline.extract(args), args.skip_train, args.skip_test,
        )
