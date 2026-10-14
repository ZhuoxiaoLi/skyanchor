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
import math

import torch
torch.set_float32_matmul_precision('highest')
import torch.nn.functional as F
import os
import json
import gc
import shutil
from random import randint
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render, network_gui_ws
import sys
from scene import Scene, GaussianModel
from scene.dataset_readers import (
    load_ground_cameras,
    load_ground_data,
    load_ground_point_cloud,
)
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import ImageFilter
import torchvision.transforms as transforms
import numpy as np
from utils.corr_init_v2 import init_gaussians_with_corr

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def clone_arg_group(group):
    return Namespace(**vars(group).copy())


def ensure_output_root(model_path):
    if model_path:
        os.makedirs(model_path, exist_ok=True)
        return model_path

    if os.getenv('OAR_JOB_ID'):
        unique_str = os.getenv('OAR_JOB_ID')
    else:
        unique_str = str(uuid.uuid4())
    root_path = os.path.join("./output/", unique_str)
    os.makedirs(root_path, exist_ok=True)
    return root_path


def resolve_sparse_ply_path(source_path, ply_name):
    ply_name = ply_name or "dense_v2.ply"
    if os.path.isabs(ply_name):
        return ply_name
    if os.path.dirname(ply_name):
        return os.path.abspath(os.path.join(source_path, ply_name))
    return os.path.join(source_path, "sparse", "0", ply_name)


def get_dense_init_ply_name(dataset, ground=False):
    if ground:
        name = getattr(dataset, "ground_dense_init_ply_name", "")
        if name:
            return name
    return getattr(dataset, "dense_init_ply_name", "dense_v2.ply") or "dense_v2.ply"


def get_dense_init_ply_path(source_path, dataset, ground=False):
    return resolve_sparse_ply_path(source_path, get_dense_init_ply_name(dataset, ground=ground))


def get_camera_alpha_mask(camera, device=None, dtype=None):
    alpha_mask = getattr(camera, "alpha_mask", None)
    if alpha_mask is None:
        alpha_mask = getattr(camera, "gt_alpha_mask", None)

    if alpha_mask is None:
        image = getattr(camera, "original_image", None)
        if image is not None:
            height, width = image.shape[-2:]
            mask_device = image.device
            mask_dtype = image.dtype
        else:
            height = getattr(camera, "image_height")
            width = getattr(camera, "image_width")
            mask_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mask_dtype = torch.float32

        if device is not None:
            mask_device = device if isinstance(device, torch.device) else torch.device(device)
        if dtype is not None:
            mask_dtype = dtype
        return torch.ones((1, height, width), device=mask_device, dtype=mask_dtype)

    if alpha_mask.dim() == 2:
        alpha_mask = alpha_mask.unsqueeze(0)
    if device is not None:
        alpha_mask = alpha_mask.to(device)
    if dtype is not None:
        alpha_mask = alpha_mask.to(dtype=dtype)
    return alpha_mask.clamp(0.0, 1.0)


def apply_camera_alpha(image, camera):
    alpha_mask = get_camera_alpha_mask(camera, device=image.device, dtype=image.dtype)
    return image * alpha_mask, alpha_mask


def photometric_loss(image, camera, lambda_dssim):
    """Use the same pixel-masked L1 and SSIM objective in every stage."""
    image_masked, alpha_mask = apply_camera_alpha(image, camera)
    gt_image_masked = camera.original_image.to(image.device) * alpha_mask
    Ll1 = l1_loss(image_masked, gt_image_masked)
    ssim_value = fast_ssim(image_masked.unsqueeze(0), gt_image_masked.unsqueeze(0))
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
    return loss, Ll1, image_masked, gt_image_masked


def filter_visibility_by_alpha(visibility_filter, gaussian_alpha_mask):
    if gaussian_alpha_mask is None:
        return visibility_filter
    if visibility_filter.dtype == torch.bool:
        return visibility_filter & gaussian_alpha_mask
    visible_ids = visibility_filter.reshape(-1)
    return visible_ids[gaussian_alpha_mask[visible_ids]]


def get_masked_camera_image(camera, device=None):
    image = camera.original_image
    if device is not None:
        image = image.to(device)
    alpha_mask = get_camera_alpha_mask(camera, device=image.device, dtype=image.dtype)
    return image * alpha_mask


def get_camera_edges_loss(camera):
    masked_image = get_masked_camera_image(camera).detach().cpu()
    edges_loss = get_edges(masked_image).squeeze().cuda()
    edge_min = edges_loss.min()
    edge_range = edges_loss.max() - edge_min
    if float(edge_range.item()) > 1e-8:
        edges_loss_norm = (edges_loss - edge_min) / edge_range
    else:
        edges_loss_norm = torch.zeros_like(edges_loss)

    alpha_mask = get_camera_alpha_mask(camera, device="cuda", dtype=edges_loss_norm.dtype).squeeze()
    if alpha_mask.shape != edges_loss_norm.shape:
        alpha_mask = F.interpolate(
            alpha_mask[None, None],
            size=edges_loss_norm.shape,
            mode="nearest",
        ).squeeze()
    return (edges_loss_norm * alpha_mask).cpu()


def maybe_run_dense_initialization(dataset, opt, scene, gaussians, args):
    dense_ply_path = get_dense_init_ply_path(dataset.source_path, dataset)
    use_unfiltered_init = os.path.basename(dense_ply_path) == "dense_v2_unfiltered.ply"
    custom_init_ply = getattr(dataset, "init_ply_path", "")
    if custom_init_ply:
        print(f"[Skip RoMa Init] Using custom init point cloud: {custom_init_ply}")
        return custom_init_ply

    pair_file_path = opt.pair_file if opt.pair_file else os.path.join(dataset.source_path, "pair.txt")
    pair_file_to_use = pair_file_path if os.path.exists(pair_file_path) else None

    if not os.path.exists(dense_ply_path):
        with torch.no_grad():
            n_splats_at_init = len(gaussians.get_xyz)
            init_gaussians_with_corr(
                gaussians, scene, opt, device="cuda", verbose=False,
                roma_model=None, pair_file=pair_file_to_use,
                use_unfiltered_init=use_unfiltered_init,
                init_scale_factor=dataset.init_scale_factor,
                init_opacity=dataset.init_opacity,
            )

            if not opt.add_SfM_init:
                n_splats_after_init = len(gaussians.get_xyz)
                print("N_splats_after_init:", n_splats_after_init)
                gaussians.tmp_radii = torch.zeros(gaussians._xyz.shape[0]).to("cuda")
                mask = torch.concat(
                    [
                        torch.ones(n_splats_at_init, dtype=torch.bool),
                        torch.zeros(n_splats_after_init - n_splats_at_init, dtype=torch.bool),
                    ],
                    axis=0,
                )
                gaussians.prune_points(mask)
    else:
        print(f"[Skip RoMa Init] Found existing dense point cloud at {dense_ply_path}")

    return dense_ply_path


def ensure_ground_dense_init_ply(ground_source_path, ground_images, dataset, opt, args):
    """Return the configured ground initialization PLY path. If missing, create a temporary Scene and run RoMa initialization."""
    dense_ply_path = get_dense_init_ply_path(ground_source_path, dataset, ground=True)
    if os.path.exists(dense_ply_path):
        print(f"[Ground RoMa] Dense point cloud already exists at {dense_ply_path}")
        return dense_ply_path

    sparse_dir = os.path.join(ground_source_path, "sparse", "0")
    has_cam = any(os.path.exists(os.path.join(sparse_dir, f))
                  for f in ("cameras.bin", "cameras.txt"))
    has_img = any(os.path.exists(os.path.join(sparse_dir, f))
                  for f in ("images.bin", "images.txt"))
    if not (has_cam and has_img):
        raise FileNotFoundError(
            f"[Ground RoMa] Missing cameras/images in {sparse_dir}/ (need cameras.{{bin,txt}} and images.{{bin,txt}})")

    print(f"\n{'='*60}")
    print(f"[Ground RoMa] Running RoMa dense initialization for ground data at {ground_source_path}")
    print(f"{'='*60}")

    prep_dir = os.path.join(ensure_output_root(args.model_path), "_ground_roma_prepare")
    os.makedirs(prep_dir, exist_ok=True)

    prep_dataset = clone_arg_group(dataset)
    prep_dataset.source_path = ground_source_path
    prep_dataset.images = ground_images
    prep_dataset.model_path = prep_dir
    prep_dataset.init_ply_path = ""
    prep_dataset.dense_init_ply_name = get_dense_init_ply_name(dataset, ground=True)
    prep_dataset.ground_dense_init_ply_name = ""

    prep_opt = clone_arg_group(opt)
    ground_pair_file = getattr(opt, 'ground_pair_file', '')
    if ground_pair_file and os.path.exists(ground_pair_file):
        prep_opt.pair_file = ground_pair_file
        print(f"[Ground RoMa] Using ground pair file: {ground_pair_file}")
    else:
        prep_opt.pair_file = ""

    gaussians = GaussianModel(prep_dataset.sh_degree, prep_opt.optimizer_type)
    scene = Scene(prep_dataset, gaussians)
    gaussians.training_setup(prep_opt)
    maybe_run_dense_initialization(prep_dataset, prep_opt, scene, gaussians, args)

    del scene
    del gaussians
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(prep_dir, ignore_errors=True)

    if not os.path.exists(dense_ply_path):
        raise FileNotFoundError(f"[Ground RoMa] Failed to generate dense init point cloud: {dense_ply_path}")

    print(f"[Ground RoMa] Finished, saved to {dense_ply_path}\n")
    return dense_ply_path


def apply_explicit_aerial_ground_length(opt, dataset) -> None:
    """Use positive aerial and ground stage lengths to set ground_start_iter and iterations."""
    gsp = getattr(dataset, "ground_source_path", "") or ""
    a = int(getattr(opt, "aerial_stage_iterations", 0) or 0)
    g = int(getattr(opt, "ground_stage_iterations", 0) or 0)
    if gsp.strip() and a > 0 and g > 0:
        opt.ground_start_iter = a
        opt.iterations = a + g


def apply_explicit_aerial_ground_length_to_ns(ns) -> None:
    gsp = getattr(ns, "ground_source_path", "") or ""
    a = int(getattr(ns, "aerial_stage_iterations", 0) or 0)
    g = int(getattr(ns, "ground_stage_iterations", 0) or 0)
    if gsp.strip() and a > 0 and g > 0:
        ns.ground_start_iter = a
        ns.iterations = a + g


def compute_densify_window_and_params(
    opt,
    *,
    has_ground: bool,
    ground_start_iter: int,
    total_iterations: int,
    stage2_active: bool,
    stage3_active: bool,
):
    """Use shared stage-local schedules; joint training has no density updates."""
    if stage3_active:
        return int(total_iterations), int(total_iterations), 0, 0, 0.0

    offset = int(ground_start_iter) if stage2_active and has_ground else 0
    stage_end = int(opt.iterations)
    if has_ground and ground_start_iter > 0:
        stage_end = int(opt.iterations) if stage2_active else int(ground_start_iter)
    return (offset + int(opt.densify_from_iter),
            min(offset + int(opt.densify_until_iter), stage_end),
            int(opt.densification_interval), int(opt.opacity_reset_interval),
            float(opt.densify_grad_threshold))


def sample_densification_views(cameras, edges, remaining_indices, count):
    """Sample across the whole sequence, keeping each camera paired with its edge map."""
    selected_cameras, selected_edges = [], []
    for _ in range(count):
        if not remaining_indices:
            remaining_indices.extend(range(len(cameras)))
        index = remaining_indices.pop(randint(0, len(remaining_indices) - 1))
        selected_cameras.append(cameras[index])
        selected_edges.append(edges[index])
    return selected_cameras, selected_edges


def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from, websockets, args, run_label=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    train_cameras_for_alpha = scene.getTrainCameras()
    alpha_cameras = [cam for cam in train_cameras_for_alpha if getattr(cam, "has_alpha_mask", False)]
    if alpha_cameras:
        sample_ratios = []
        for cam in alpha_cameras[: min(16, len(alpha_cameras))]:
            alpha = get_camera_alpha_mask(cam, dtype=torch.float32)
            sample_ratios.append(float((alpha <= 0.5).float().mean().item()))
        mean_masked = sum(sample_ratios) / max(len(sample_ratios), 1)
        print(
            f"[AlphaMask] Loaded RGBA alpha for {len(alpha_cameras)}/{len(train_cameras_for_alpha)} "
            f"train cameras; sampled masked-pixel ratio={mean_masked:.3f}"
        )
    else:
        print("[AlphaMask] No RGBA alpha mask loaded; sky pixels will not be excluded by alpha.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    all_edges = []
    for view in scene.getTrainCameras():
        all_edges.append(get_camera_edges_loss(view))
    edge_view_indices = []

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    prune_cumulative = 0
    split_cumulative = 0
    progress_desc = "Training progress" if run_label is None else f"Training progress [{run_label}]"
    apply_explicit_aerial_ground_length(opt, dataset)
    ground_path_dbg = getattr(dataset, "ground_source_path", "") or ""
    has_ground_dbg = bool(str(ground_path_dbg).strip())
    g_start_dbg = int(getattr(opt, "ground_start_iter", 0) or 0)
    jf_dbg = int(getattr(opt, "joint_finetune_iterations", 0) or 0)
    if has_ground_dbg and jf_dbg > 0:
        total_iterations = int(opt.iterations) + jf_dbg
    else:
        total_iterations = int(opt.iterations)
    if has_ground_dbg:
        print(
            f"[Schedule] ground_start_iter={g_start_dbg}  "
            f"iterations(aerial+ground before joint)={opt.iterations}  "
            f"joint_finetune_iterations={jf_dbg}  total_iterations={total_iterations}"
        )
    else:
        print(f"[Schedule] single-stage, iterations={opt.iterations}")
    first_iter = 1
    progress_bar = tqdm(range(first_iter, total_iterations + 1), desc=progress_desc)


    bg = torch.rand((3), device="cuda") if opt.random_background else background


    maybe_run_dense_initialization(dataset, opt, scene, gaussians, args)

    ground_source_path = getattr(dataset, "ground_source_path", "") or ""
    if not str(ground_source_path).strip():
        ground_source_path = ""
    ground_start_iter = int(getattr(opt, "ground_start_iter", 0) or 0)
    joint_finetune_iters = int(getattr(opt, "joint_finetune_iterations", 0) or 0)
    ground_cameras = None
    ground_all_edges = None
    ground_report_train_cameras = None
    ground_report_test_cameras = None
    ground_report_extent = None
    gaussians.configure_position_schedule(opt, ground_start_iter if ground_source_path else opt.position_lr_max_steps)
    stage2_active = False
    stage3_active = False
    aerial_cache = None
    all_cameras_combined = None

    if ground_source_path:
        ground_images_dir = getattr(dataset, 'ground_images', 'images')
        ground_report_train_cameras, ground_report_test_cameras, ground_report_extent = load_ground_cameras(
            ground_source_path,
            ground_images_dir,
            1.0,
            dataset,
        )
        aerial_all = list(scene.getTrainCameras()) + list(scene.getTestCameras())
        aerial_max_uid = (max(c.uid for c in aerial_all) + 1) if aerial_all else 0
        for i, cam in enumerate(list(ground_report_train_cameras) + list(ground_report_test_cameras)):
            cam.uid = aerial_max_uid + i
        print(
            f"[Ground Eval] Loaded ground cameras for reporting from Stage1: "
            f"{len(ground_report_train_cameras)} train / {len(ground_report_test_cameras)} test"
        )

    for iteration in range(first_iter, total_iterations + 1):

        if (ground_source_path and not stage2_active
                and iteration == ground_start_iter + 1):
            stage2_active = True
            prune_cumulative = split_cumulative = 0
            n_aerial = gaussians.get_xyz.shape[0]
            if ground_start_iter not in saving_iterations:
                scene.save(ground_start_iter)
            aerial_cache = gaussians.cache_parameters()
            gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            scene.gaussians = gaussians
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n{'='*60}")
            print(f"[Stage2] Switching to ground stage at iter {iteration}")
            print(f"[Stage2] Aerial Gaussians cached on CPU, excluded from rendering and training: {n_aerial}")

            ground_images_dir = getattr(dataset, 'ground_images', 'images')
            ensure_ground_dense_init_ply(
                ground_source_path, ground_images_dir, dataset, opt, args)
            if ground_report_train_cameras is None:
                ground_cameras, ground_test_cameras, ground_pcd, ground_extent = load_ground_data(
                    ground_source_path, ground_images_dir,
                    1.0, dataset,
                )
            else:
                ground_cameras = ground_report_train_cameras
                ground_test_cameras = ground_report_test_cameras
                ground_extent = ground_report_extent
                ground_pcd = load_ground_point_cloud(ground_source_path, dataset)
            all_ground_cams = list(ground_cameras) + list(ground_test_cameras)
            aerial_all = list(scene.getTrainCameras()) + list(scene.getTestCameras())
            aerial_max_uid = (max(c.uid for c in aerial_all) + 1) if aerial_all else 0
            for i, cam in enumerate(all_ground_cams):
                cam.uid = aerial_max_uid + i

            if ground_test_cameras:
                print(f"[Stage2] Ground test cameras kept as separate eval split: {len(ground_test_cameras)}")

            print(f"[Stage2] Ground train cameras: {len(ground_cameras)}, "
                  f"Ground points: {len(ground_pcd.points)}")

            gaussians.create_from_pcd(
                ground_pcd, ground_extent, dataset.init_scale_factor, dataset.init_opacity,
            )
            gaussians.training_setup(opt)
            gaussians.configure_position_schedule(opt, opt.iterations - ground_start_iter)
            print(f"[Stage2] Independent ground model: {len(gaussians.get_xyz)} Gaussians, SH degree 0, fresh optimizer")

            viewpoint_stack = ground_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

            ground_all_edges = []
            for view in ground_cameras:
                ground_all_edges.append(get_camera_edges_loss(view))
            all_edges = ground_all_edges.copy()
            edge_view_indices = []

            ema_loss_for_log = 0.0
            ema_psnr_for_log = 0.0
            print(f"[Stage2] Ground stage started at iter {iteration}")
            print(f"{'='*60}\n")

        if (ground_source_path and stage2_active and not stage3_active
                and joint_finetune_iters > 0
                and iteration == opt.iterations + 1):
            stage3_active = True
            prune_cumulative = split_cumulative = 0
            if opt.iterations not in saving_iterations:
                scene.save(opt.iterations)
            gaussians.merge_cached_parameters(
                aerial_cache, opt, dataset.init_scale_factor, dataset.init_opacity,
            )
            joint_num_gaussians = len(gaussians.get_xyz)
            aerial_cache = None
            gaussians.configure_position_schedule(opt, joint_finetune_iters)
            all_cameras_combined = list(scene.getTrainCameras()) + list(ground_cameras)

            aerial_edges = []
            for view in scene.getTrainCameras():
                aerial_edges.append(get_camera_edges_loss(view))
            all_edges = aerial_edges + ground_all_edges
            viewpoint_stack = list(all_cameras_combined)
            viewpoint_indices = list(range(len(viewpoint_stack)))
            edge_view_indices = []

            ema_loss_for_log = 0.0
            ema_psnr_for_log = 0.0

            print(f"\n{'='*60}")
            print(f"[Stage3] Joint finetune started at iter {iteration}")
            print(f"[Stage3] All Gaussians trainable: {joint_num_gaussians}")
            print(f"[Stage3] Scale reset to merged RMS-3NN * {dataset.init_scale_factor:g}; opacity={dataset.init_opacity:g}")
            print("[Stage3] Fixed Gaussian count; no densification, pruning, or periodic opacity resets")
            print(f"[Stage3] Combined cameras: {len(all_cameras_combined)} "
                  f"(aerial {len(scene.getTrainCameras())} + ground {len(ground_cameras)})")
            print(f"[Stage3] Will run for {joint_finetune_iters} iterations "
                  f"(until iter {total_iterations})")
            print(f"{'='*60}\n")

            progress_bar.total = total_iterations
            progress_bar.refresh()

        if stage3_active:
            active_cams = all_cameras_combined
        elif stage2_active:
            active_cams = ground_cameras
        else:
            active_cams = scene.getTrainCameras()

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(active_cams):
                cam = active_cams[network_gui_ws.curr_id]
                net_image = render(cam, gaussians, pipe, background, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()

        stage_offset = opt.iterations if stage3_active else (ground_start_iter if stage2_active else 0)
        stage_iteration = iteration - stage_offset
        gaussians.update_learning_rate(stage_iteration - 1)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if stage_iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            if stage3_active:
                viewpoint_stack = list(all_cameras_combined)
            elif stage2_active:
                viewpoint_stack = ground_cameras.copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        loss, Ll1, image_masked, gt_image_masked = photometric_loss(image, viewpoint_cam, opt.lambda_dssim)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            prn_this = 0
            split_this = 0
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            batch_psnr = float(psnr(image_masked.detach(), gt_image_masked.detach()).mean().item())
            ema_psnr_for_log = 0.4 * batch_psnr + 0.6 * ema_psnr_for_log

            # Log and save
            validation_configs = None
            if ground_report_train_cameras is not None:
                ground_validation = (
                    {'name': 'test_ground', 'cameras': ground_report_test_cameras},
                    {'name': 'train_ground', 'cameras': ground_report_train_cameras},
                )
                aerial_validation = (
                    {'name': 'test_aerial', 'cameras': scene.getTestCameras()},
                    {'name': 'train_aerial', 'cameras': scene.getTrainCameras()},
                )
                validation_configs = ground_validation if stage2_active and not stage3_active else aerial_validation + ground_validation
            cull_thr = float(getattr(opt, "cull_opacity", 0.003))

            hg = bool(ground_source_path and str(ground_source_path).strip())
            densify_from, densify_until, d_int, o_int, g_thr = compute_densify_window_and_params(
                opt,
                has_ground=hg,
                ground_start_iter=ground_start_iter,
                total_iterations=total_iterations,
                stage2_active=bool(stage2_active and not stage3_active),
                stage3_active=bool(stage3_active),
            )

            if densify_from < iteration < densify_until:
                gaussians.add_densification_stats_abs(viewspace_point_tensor, visibility_filter)

            # Consume gradients before densification can replace parameter tensors.
            gaussians.optimizer_step(radii > 0)

            if densify_from < iteration < densify_until:
                if d_int > 0 and stage_iteration % d_int == 0:
                    active_cam_list = active_cams
                    num_cams = args.cams
                    if args.cams == -1 or (stage_iteration % 6000 == 400 and stage_iteration < 30000):
                        num_cams = len(active_cam_list)
                    camlist, edge_losses = sample_densification_views(
                        active_cam_list, all_edges, edge_view_indices, num_cams)
                    gaussian_importance = compute_edge_score(camlist, edge_losses, gaussians, pipe, bg)

                    startI = densify_from
                    endI = densify_until - 500
                    rate = (iteration - startI) / max(endI - startI, 1)
                    cap = opt.budget
                    if rate >= 1.0:
                        b_alloc = cap
                    else:
                        b_alloc = int(math.sqrt(max(rate, 0.0)) * cap)

                    stats = gaussians.densify_and_prune_Improved(
                        gaussian_importance, cull_thr, b_alloc, opt, stage_iteration
                    )
                    prn_this += stats["pruned"]
                    split_this += stats["split"]
                    if getattr(opt, "densify_debug", False):
                        st = (
                            "Stage3-joint"
                            if stage3_active
                            else ("Stage2-ground" if stage2_active else "Stage1-aerial")
                        )
                        n_gs = int(gaussians.get_xyz.shape[0])
                        n_tr = n_gs - int(
                            getattr(gaussians, "_n_frozen", 0) or 0
                        )
                        print(
                            f"[Densify-debug] {st}  iter={iteration}  d_int={d_int}  "
                            f"N_GS={n_gs}  N_trainable={n_tr}  cap={cap}  b_alloc={b_alloc}  "
                            f"window=({densify_from}, {densify_until}) split={stats['split']} "
                            f"pruned={stats['pruned']}"
                        )

                if o_int > 0 and stage_iteration % o_int == 0:
                    gaussians.reset_opacity(opt.opacity_reset_max)
                    print(f"[Opacity reset] iter={iteration} max_opacity={opt.opacity_reset_max:g}", flush=True)

            if stage3_active and len(gaussians.get_xyz) != joint_num_gaussians:
                raise RuntimeError("The Gaussian count changed during joint training")

            prune_cumulative += prn_this
            split_cumulative += split_this
            if iteration % 10 == 0 or iteration == total_iterations:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "PSNR": f"{ema_psnr_for_log:.3f}",
                        "N_GS": f"{gaussians.get_xyz.shape[0]}",
                        "splitN": f"{split_cumulative}",
                        "pruneN": f"{prune_cumulative}",
                    },
                    refresh=False,
                )
                progress_bar.update(iteration - progress_bar.n)
            if iteration == total_iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                validation_configs=validation_configs,
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

        if ground_source_path and iteration in (ground_start_iter, opt.iterations):
            # Release the previous render graph before allocating the next stage.
            del loss, Ll1, image, image_masked, render_pkg, viewspace_point_tensor
            if websockets:
                net_image = None

    if tb_writer:
        tb_writer.close()
    del scene
    del gaussians
    torch.cuda.empty_cache()


def get_edges(image):
    image = image.detach().cpu()
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    return image_edges_tensor


def normalize(value_tensor):
    value_tensor[value_tensor.isnan()] = 0
    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = valid_value / torch.mean(valid_value)

    return ret_value


def compute_edge_score(camlist, edge_losses, gaussians, pipe, bg):
    num_points = len(gaussians.get_xyz)
    gaussian_importance = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    visibility_filter_all = torch.zeros(num_points, device="cuda", dtype=bool)
    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        pixel_weights = edge_losses[view].cuda()
        alpha_mask = get_camera_alpha_mask(my_viewpoint_cam, device="cuda", dtype=pixel_weights.dtype).squeeze()
        if alpha_mask.shape != pixel_weights.shape:
            alpha_mask = F.interpolate(
                alpha_mask[None, None],
                size=pixel_weights.shape,
                mode="nearest",
            ).squeeze()
        pixel_weights = pixel_weights * alpha_mask
        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights=pixel_weights)

        loss_accum = normalize(render_pkg["accum_weights"])
        alpha_gaussian_mask = render_pkg["accum_weights"].detach().reshape(-1) > 0
        visibility_filter = filter_visibility_by_alpha(
            render_pkg["visibility_filter"].detach(),
            alpha_gaussian_mask,
        )

        gaussian_importance[visibility_filter] += loss_accum[visibility_filter] / len(camlist)
        visibility_filter_all[visibility_filter] = True

    return gaussian_importance


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    validation_configs=None,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if validation_configs is None:
            validation_configs = (
                {'name': 'test', 'cameras': scene.getTestCameras()},
                {'name': 'train', 'cameras': scene.getTrainCameras()},
            )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test= 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    image, alpha_mask = apply_camera_alpha(image, viewpoint)
                    gt_image = gt_image * alpha_mask
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,30000,50000,99980,200000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cams", type=int, default=6)
    parser.add_argument("--websockets", action='store_true', default=False)
    from prepare_inputs import add_arguments, prepare_training_inputs
    add_arguments(parser)
    args = parser.parse_args(sys.argv[1:])
    args.model_path = ensure_output_root(args.model_path)
    prepare_training_inputs(args)
    if args.prepare_only:
        print("Training inputs prepared successfully.")
        raise SystemExit(0)
    apply_explicit_aerial_ground_length_to_ns(args)
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if getattr(args, 'ground_source_path', '') and args.ground_source_path.strip():
        gs_iter = getattr(args, 'ground_start_iter', 30000)
        args.test_iterations.append(gs_iter)
        if gs_iter not in args.save_iterations:
            args.save_iterations.append(gs_iter)
        jf_iters = getattr(args, 'joint_finetune_iterations', 0)
        if jf_iters > 0:
            jf_end = args.iterations + jf_iters
            args.test_iterations.append(jf_end)
            if args.iterations not in args.save_iterations:
                args.save_iterations.append(args.iterations)
            if jf_end not in args.save_iterations:
                args.save_iterations.append(jf_end)

    args.test_iterations = sorted(set(args.test_iterations))
    args.model_path = ensure_output_root(args.model_path)

    with open(os.path.join(args.model_path, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    training(dataset, opt, pipe, args.test_iterations, args.save_iterations,
             args.debug_from, args.websockets, args)

    # All done
    print("\nTraining complete.")
