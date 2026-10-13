#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import gc
import os
import random
import zlib

import numpy as np
import torch
from tqdm import tqdm

from dataloaders.image_dataset import ImageDataset
from poses.feature_detector import Detector
from poses.colmap_export import (
    export_final_ba_colmap_text,
    export_pose_snapshots_colmap_text,
)
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer
from poses.triangulator import Triangulator
from scene.keyframe import Keyframe
from scene.pose_scene import PoseScene
from utils import (
    get_global_profiler,
)


def seed_random_generators(seed: int):
    """Seed every sampler used by pose estimation and robust matching."""
    seed = int(seed) & 0x7FFFFFFF
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except (ImportError, AttributeError):
        pass
    return seed






def update_scene_focal(scene_model: PoseScene, keyframes: list[Keyframe], focal: float):
    focal_tensor = torch.tensor([focal], device="cuda").float()
    scene_model.f = focal
    for keyframe in keyframes:
        if getattr(keyframe, "f", None) is not None:
            keyframe.f = focal_tensor
    return focal_tensor


def get_final_ba_export_dir(args):
    if args.final_submap_ba_export_dir:
        return args.final_submap_ba_export_dir
    return os.path.join(args.model_path, "final_ba_colmap")


def update_profiler_frame_meta(profiler, **meta):
    frame_meta = getattr(profiler, "_frame_meta", None)
    if frame_meta is not None:
        frame_meta.update(meta)


def export_final_ba_sparse_model(scene_model: PoseScene, sparse_payloads, args):
    if len(scene_model.keyframes) == 0:
        return None
    if not args.final_submap_ba and not args.final_submap_ba_export_dir:
        return None

    export_stats = export_final_ba_colmap_text(
        get_final_ba_export_dir(args),
        scene_model.keyframes,
        sparse_payloads,
    )
    print(
        "Saved final BA COLMAP text model: "
        f"path={export_stats['path']}, "
        f"images={export_stats['num_images']}, "
        f"points3D={export_stats['num_points3D']}, "
        f"observations={export_stats['num_observations']}"
    )
    return export_stats


def export_online_pose_sparse_model(scene_model: PoseScene, args):
    export_dir = getattr(args, "online_pose_export_dir", "")
    if not export_dir or len(scene_model.keyframes) == 0:
        return None
    export_stats = export_pose_snapshots_colmap_text(
        export_dir,
        scene_model.keyframes,
    )
    print(
        "Saved pre-BA online-pose COLMAP text model: "
        f"path={export_stats['path']}, "
        f"images={export_stats['num_images']}, "
        f"points3D={export_stats['num_points3D']}"
    )
    return export_stats


def run_anchored_ba(
    pose_initializer: PoseInitializer,
    scene_model: PoseScene,
    keyframes: list[Keyframe],
    args,
    label: str,
    return_point_colors: bool = False,
    num_pts: int | None = None,
    iters: int | None = None,
    candidate_multiplier: float | None = None,
    return_colmap_payload: bool = True,
    require_tail_connectivity: bool = True,
    strict_graph: bool = True,
    causal_front_only: bool = False,
    optimize_focal: bool | None = None,
):
    if len(keyframes) == 0:
        print(f"{label} skipped: no keyframes.")
        return None

    if causal_front_only:
        leaked_tail = [
            keyframe.info.get("name", keyframe.index)
            for keyframe in keyframes
            if keyframe.info.get("is_submap_tail_anchor", False)
        ]
        leaked_non_front_anchor = [
            keyframe.info.get("name", keyframe.index)
            for keyframe in keyframes
            if keyframe.info.get("anchor_Rt", None) is not None
            and not keyframe.info.get("is_submap_seed_anchor", False)
        ]
        if leaked_tail or leaked_non_front_anchor:
            raise RuntimeError(
                "Causal front-anchor BA input contains future/non-front anchors "
                "before any anchor pose was read: "
                f"tail={leaked_tail}, non_front_fixed={leaked_non_front_anchor}."
            )

    anchor_keyframes = [
        keyframe
        for keyframe in keyframes
        if keyframe.info.get("anchor_Rt", None) is not None
    ]
    for keyframe in anchor_keyframes:
        keyframe.set_Rt(keyframe.info["anchor_Rt"])

    final_ba_stats = pose_initializer.finalize_submap_ba(
        keyframes,
        args.final_submap_ba_num_pts if num_pts is None else num_pts,
        args.final_submap_ba_iters if iters is None else iters,
        optimize_focal=(
            args.final_submap_ba_optimize_focal
            if optimize_focal is None
            else optimize_focal
        ),
        candidate_multiplier=(
            args.final_submap_ba_candidate_multiplier
            if candidate_multiplier is None
            else candidate_multiplier
        ),
        return_point_colors=return_point_colors,
        return_track_groups=return_point_colors,
        return_colmap_payload=return_colmap_payload,
        require_tail_connectivity=require_tail_connectivity,
        strict_graph=strict_graph,
        causal_front_only=causal_front_only,
    )
    if final_ba_stats is None:
        if len(anchor_keyframes) > 0:
            scene_model.refresh_pose_cache()
        print(f"{label} skipped: not enough anchor-constrained tracks.")
        return None

    scene_model.refresh_pose_cache()

    print(
        f"{label}: "
        f"backend={final_ba_stats.get('backend', 'miniba')}, "
        f"solver={final_ba_stats.get('linear_solver', 'default')}, "
        f"residual={final_ba_stats['residual']:.4f}, "
        f"movable={final_ba_stats['n_movable']}, "
        f"fixed={final_ba_stats['n_fixed']}, "
        f"selected_tracks={final_ba_stats['selected_tracks']}, "
        f"total_tracks={final_ba_stats.get('candidate_tracks', final_ba_stats['selected_tracks'])}, "
        f"anchor_tracks={final_ba_stats['selected_anchor_tracks']}/"
        f"{final_ba_stats['candidate_anchor_tracks']}, "
        f"non_anchor_tracks={final_ba_stats.get('selected_non_anchor_tracks', 0)}/"
        f"{final_ba_stats.get('candidate_non_anchor_tracks', 0)}, "
        f"preferred_tracks={final_ba_stats.get('selected_preferred_tracks', 0)}, "
        f"depth_kept={final_ba_stats.get('selected_depth_kept_tracks', final_ba_stats['selected_tracks'])}, "
        f"depth_filtered={final_ba_stats.get('selected_depth_filtered_tracks', 0)}, "
        f"front_anchors_min={final_ba_stats.get('min_reachable_front_anchors', 0)}, "
        f"front_baseline_min={final_ba_stats.get('min_reachable_front_baseline', 0.0):.4g}, "
        f"mean_obs={final_ba_stats['selected_mean_obs']:.2f}, "
        f"obs={final_ba_stats.get('selected_obs_distribution', 'n/a')}, "
        f"mean_score={final_ba_stats.get('selected_mean_score', 0.0):.3f}, "
        f"focal={final_ba_stats['focal']:.2f}, "
        f"usable={final_ba_stats.get('solver_usable', False)}, "
        f"termination={final_ba_stats.get('termination_type', 'n/a')}"
    )
    if final_ba_stats.get("solver_report"):
        print(f"{label} solver: {final_ba_stats['solver_report']}")
    return final_ba_stats


def remap_submap_keyframes_to_global(keyframes: list[Keyframe], start_index: int):
    index_map = {
        keyframe.index: start_index + local_index
        for local_index, keyframe in enumerate(keyframes)
    }
    for local_index, keyframe in enumerate(keyframes):
        if keyframe.desc_kpts is not None:
            keyframe.desc_kpts.matches = {
                index_map[other_index]: matches
                for other_index, matches in keyframe.desc_kpts.matches.items()
                if other_index in index_map
            }
        keyframe.info["submap_local_index"] = local_index
        keyframe.index = start_index + local_index


def append_submap_keyframes(
    global_scene_model: PoseScene,
    submap_keyframes: list[Keyframe],
    f: torch.Tensor,
):
    start_index = len(global_scene_model.keyframes)
    remap_submap_keyframes_to_global(submap_keyframes, start_index)
    for keyframe in submap_keyframes:
        global_scene_model.add_keyframe(keyframe, f)


def release_submap_resources(
    local_scene_model: PoseScene,
    drop_dense: bool = False,
    label: str = "",
    verbose: bool = False,
):
    """
    Free GPU memory accumulated during a submap pass.

    - Always: clear ``desc_kpts.matches`` on every local keyframe (cross-frame
      pairwise matches stored on each ``DescribedKeypoints`` are only consumed by
      ``build_problem`` during the same submap's bootstrap / incremental / final
      BA; once ``finalize_submap_ba`` returned, they are dead weight on GPU).
      Then runs ``gc.collect()`` and ``torch.cuda.empty_cache()``.
    - ``drop_dense=True``: in addition, offload every local keyframe to CPU
      (images and desc_kpts tensors). Use this in ``--pose_debug_only`` runs where the
      keyframes are kept around only for the viser viewer / final BA export
      and never accessed on GPU again.
    """
    cuda_available = torch.cuda.is_available()
    if verbose and cuda_available:
        before_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        before_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    keyframes = list(getattr(local_scene_model, "keyframes", []))
    n_match_entries = 0
    for keyframe in keyframes:
        desc_kpts = getattr(keyframe, "desc_kpts", None)
        if desc_kpts is not None:
            n_match_entries += len(getattr(desc_kpts, "matches", {}) or {})
            desc_kpts.matches = {}
        if drop_dense:
            try:
                keyframe.to("cpu")
            except Exception:
                pass
            if desc_kpts is not None:
                try:
                    desc_kpts.to("cpu")
                except Exception:
                    pass
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()
    if verbose and cuda_available:
        after_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        after_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        prefix = f"[mem cleanup {label}] " if label else "[mem cleanup] "
        print(
            f"{prefix}cleared {n_match_entries} match entries from "
            f"{len(keyframes)} keyframes; drop_dense={drop_dense}; "
            f"alloc {before_alloc:.0f}->{after_alloc:.0f} MB, "
            f"reserved {before_reserved:.0f}->{after_reserved:.0f} MB"
        )


def run_anchor_submap_pose_pass(
    dataset: ImageDataset,
    width: int,
    height: int,
    args,
    matcher: Matcher,
    pose_initializer: PoseInitializer,
    triangulator: Triangulator,
    detector: Detector,
    scene_model: PoseScene,
    focal: float,
    pose_debugger=None,
):
    f = torch.tensor([focal], device="cuda").float()
    submap_ranges = dataset.anchor_submap_ranges
    if not submap_ranges:
        raise ValueError("No anchored submap ranges are available.")
    final_ba_payloads = []
    pbar = tqdm(range(0, len(dataset)), desc="Pose submaps")
    prof = get_global_profiler()

    for submap_id, (submap_start, submap_end) in enumerate(submap_ranges):
        if hasattr(dataset, "image_name_list") and submap_end > submap_start:
            first_submap_name = dataset.image_name_list[submap_start]
            last_submap_name = dataset.image_name_list[submap_end - 1]
            seed_key = f"{first_submap_name}|{last_submap_name}"
        else:
            seed_key = f"{submap_start}|{submap_end}"
        submap_seed = seed_random_generators(
            int(args.random_seed) + zlib.crc32(seed_key.encode("utf-8"))
        )
        print(
            f"[reproducibility] submap={submap_id}, seed={submap_seed}, "
            f"key={seed_key}"
        )
        submap_len = submap_end - submap_start
        bootstrap_count = min(args.num_keyframes_miniba_bootstrap, submap_len)
        seed_count = min(args.num_seed_anchor_frames, bootstrap_count)

        prof.begin_submap(submap_id, n_input_frames=submap_len, bootstrap_count=bootstrap_count)
        local_scene_model = PoseScene(width, height, args, matcher)
        local_scene_model.f = focal
        bootstrap_keyframe_dicts = []
        bootstrap_desc_kpts = []
        local_n_keyframes = 0
        prev_desc_kpts = None
        metrics = {}
        online_anchor_ba_tail_flushed = False

        def run_online_front_anchor_ba(reason: str, local_scene_model):
            with prof.stage("online_anchor_ba.total"):
                stats = run_anchored_ba(
                    pose_initializer,
                    local_scene_model,
                    local_scene_model.keyframes,
                    args,
                    f"Online front-anchor BA submap {submap_id} ({reason})",
                    num_pts=args.online_anchor_ba_num_pts,
                    iters=args.online_anchor_ba_iters,
                    candidate_multiplier=args.online_anchor_ba_candidate_multiplier,
                    return_colmap_payload=False,
                    require_tail_connectivity=False,
                    strict_graph=False,
                    causal_front_only=True,
                    # Online localization must keep the calibrated intrinsics
                    # used by subsequent PnP and newly constructed keyframes.
                    optimize_focal=False,
                )
                if stats is not None and stats.get("solver_usable", False):
                    # A BA pose update invalidates every cached triangulation.  Clear
                    # the whole cache before rebuilding it; update_3dpts() only
                    # overwrites points that triangulate successfully.
                    for online_ba_keyframe in local_scene_model.keyframes:
                        online_ba_keyframe.desc_kpts.clear_3D_pts()
                    for online_ba_keyframe in local_scene_model.keyframes:
                        online_ba_keyframe.update_3dpts(local_scene_model.keyframes)
                    prof.incr_counter("online_anchor_ba_completed")
                else:
                    if stats is not None:
                        print(
                            f"Online front-anchor BA submap {submap_id} ({reason}) "
                            "did not return a usable solution; cached 3D points were kept."
                        )
                    prof.incr_counter("online_anchor_ba_skipped")
            return stats

        for frameID in range(submap_start, submap_end):
            prof.begin_frame(
                frame_id=frameID,
                submap=submap_id,
                is_keyframe=False,
                is_anchor=False,
                phase="incremental",
            )
            with prof.stage("io.getnext"):
                image, info = dataset.getnext()
            with prof.stage("detector"):
                desc_kpts = detector(image)

            if (
                args.online_anchor_ba_every > 0
                and info.get("is_submap_tail_anchor", False)
                and not online_anchor_ba_tail_flushed
                and local_n_keyframes >= bootstrap_count
            ):
                run_online_front_anchor_ba("before tail anchors", local_scene_model)
                online_anchor_ba_tail_flushed = True

            if args.pose_debug_all_frames:
                should_add_keyframe = True
            elif prev_desc_kpts is None:
                should_add_keyframe = True
            else:
                with prof.stage("match.displacement"):
                    curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
                    dist = torch.norm(
                        curr_prev_matches.kpts - curr_prev_matches.kpts_other,
                        dim=-1,
                    )
                    should_add_keyframe = (
                        dist.median() > max(args.min_displacement * width, 30)
                        and len(curr_prev_matches.kpts) > args.min_num_inliers
                    )

            should_add_keyframe |= info["is_test"]
            should_add_keyframe |= info.get("is_anchor", False)
            should_add_keyframe |= local_n_keyframes < bootstrap_count

            if should_add_keyframe and local_n_keyframes < bootstrap_count:
                update_profiler_frame_meta(
                    prof,
                    phase="bootstrap_collect",
                    is_keyframe=True,
                    is_anchor=bool(info.get("is_anchor", False)),
                )
                bootstrap_keyframe_dicts.append({"image": image, "info": info})
                bootstrap_desc_kpts.append(desc_kpts)
                local_n_keyframes += 1

                if local_n_keyframes == bootstrap_count:
                    use_seeded_bootstrap = (
                        seed_count > 0
                        and all(
                            keyframe_dict["info"].get("is_submap_seed_anchor", False)
                            for keyframe_dict in bootstrap_keyframe_dicts[:seed_count]
                        )
                    )
                    if not use_seeded_bootstrap:
                        raise ValueError(
                            f"The first {seed_count} keyframes in submap {submap_id} "
                            "must be seed anchors."
                        )

                    seed_Rts = torch.stack(
                        [
                            keyframe_dict["info"]["anchor_Rt"]
                            for keyframe_dict in bootstrap_keyframe_dicts[:seed_count]
                        ],
                        dim=0,
                    )
                    with prof.stage("bootstrap.total"):
                        Rts, f, _ = pose_initializer.initialize_bootstrap_seeded(
                            bootstrap_desc_kpts, seed_Rts
                        )
                    focal = f.cpu().item()

                    with prof.stage("bootstrap.add_keyframes"):
                        for index, (keyframe_dict, keyframe_desc_kpts, Rt) in enumerate(
                            zip(bootstrap_keyframe_dicts, bootstrap_desc_kpts, Rts)
                        ):
                            if keyframe_dict["info"].get("is_submap_seed_anchor", False):
                                Rt = keyframe_dict["info"]["anchor_Rt"]
                            keyframe = Keyframe(
                                keyframe_dict["image"],
                                keyframe_dict["info"],
                                keyframe_desc_kpts,
                                Rt,
                                index,
                                f,
                                triangulator,
                                args,
                                lock_pose=(
                                    args.lock_seed_anchor_poses
                                    and keyframe_dict["info"].get("is_submap_seed_anchor", False)
                                ),
                            )
                            local_scene_model.add_keyframe(keyframe, f)

                    with prof.stage("bootstrap.update_3dpts"):
                        for keyframe in local_scene_model.keyframes:
                            keyframe.update_3dpts(local_scene_model.keyframes)
                    prof.incr_counter("bootstraps_done")
                    if pose_debugger is not None:
                        pose_debugger.update_keyframes(
                            local_scene_model.keyframes,
                            f,
                            stage="online",
                            label=f"submap {submap_id} bootstrap",
                        )


                if not info["is_test"]:
                    prev_desc_kpts = desc_kpts
                pbar.update(1)
                prof.end_frame()
                continue

            if should_add_keyframe and local_n_keyframes >= bootstrap_count:
                anchor_Rt = info.get("anchor_Rt", None)
                direct_anchor_pose = bool(
                    info.get("is_anchor", False) and anchor_Rt is not None
                )
                if direct_anchor_pose:
                    # A tail anchor already has its exact fixed pose.  Running
                    # PnP first can only create a misleading failure/retry; the
                    # final 6xN boundary matching is built explicitly before BA.
                    Rt = anchor_Rt
                else:
                    with prof.stage("kf_select"):
                        prev_keyframes = local_scene_model.get_prev_keyframes(
                            args.num_prev_keyframes_miniba_incr, True, desc_kpts
                        )
                    pose_init_Rt = local_scene_model.keyframes[-1].get_Rt()
                    with prof.stage("incremental.total"):
                        Rt = pose_initializer.initialize_incremental(
                            prev_keyframes,
                            desc_kpts,
                            local_n_keyframes,
                            info["is_test"],
                            image,
                            info.get("name", frameID),
                            pose_init_Rt=pose_init_Rt,
                        )
                    retry_num_prev = int(
                        getattr(args, "num_prev_keyframes_miniba_incr_retry", 0)
                    )
                    if (
                        Rt is None
                        and retry_num_prev > args.num_prev_keyframes_miniba_incr
                        and len(local_scene_model.keyframes)
                        > args.num_prev_keyframes_miniba_incr
                    ):
                        retry_num_prev = min(
                            retry_num_prev, len(local_scene_model.keyframes)
                        )
                        print(
                            "[incremental] retrying failed PnP with "
                            f"{retry_num_prev} reference keyframes "
                            f"(frame={info.get('name', frameID)})"
                        )
                        prof.incr_counter("incremental_retry_attempts")
                        with prof.stage("kf_select.retry"):
                            retry_prev_keyframes = local_scene_model.get_prev_keyframes(
                                retry_num_prev, True, desc_kpts
                            )
                        with prof.stage("incremental.retry"):
                            Rt = pose_initializer.initialize_incremental(
                                retry_prev_keyframes,
                                desc_kpts,
                                local_n_keyframes,
                                info["is_test"],
                                image,
                                info.get("name", frameID),
                                pose_init_Rt=pose_init_Rt,
                            )
                        prof.incr_counter(
                            "incremental_retry_recovered"
                            if Rt is not None
                            else "incremental_retry_failed"
                        )

                if Rt is not None:
                    with prof.stage("incremental.add_keyframe"):
                        keyframe = Keyframe(
                            image,
                            info,
                            desc_kpts,
                            Rt,
                            local_n_keyframes,
                            f,
                            triangulator,
                            args,
                            lock_pose=False,
                        )
                        local_scene_model.add_keyframe(keyframe)
                    with prof.stage("incremental.update_3dpts"):
                        keyframe.update_3dpts(local_scene_model.keyframes)
                    local_n_keyframes += 1
                    update_profiler_frame_meta(
                        prof,
                        is_keyframe=True,
                        is_anchor=bool(info.get("is_anchor", False)),
                    )
                    prof.incr_counter("incremental_keyframes")
                    prof.incr_counter(
                        "direct_tail_anchor_keyframes"
                        if direct_anchor_pose
                        else "incremental_pnp_keyframes"
                    )
                    n_movable = sum(
                        keyframe.info.get("anchor_Rt", None) is None
                        for keyframe in local_scene_model.keyframes
                    )
                    bootstrap_free_count = max(bootstrap_count - seed_count, 0)
                    n_new_movable = n_movable - bootstrap_free_count
                    if (
                        args.online_anchor_ba_every > 0
                        and not info.get("is_submap_tail_anchor", False)
                        and n_new_movable > 0
                        and n_new_movable % args.online_anchor_ba_every == 0
                    ):
                        run_online_front_anchor_ba(
                            f"{n_new_movable} incremental movable cameras", local_scene_model
                        )
                    if pose_debugger is not None:
                        with prof.stage("pose_debugger.update"):
                            pose_debugger.update_keyframes(
                                local_scene_model.keyframes,
                                f,
                                stage="online",
                                label=f"submap {submap_id}",
                            )

                    if not info["is_test"]:
                        prev_desc_kpts = desc_kpts
                else:
                    should_add_keyframe = False
                    prof.incr_counter("incremental_pnp_failures")

            if should_add_keyframe:
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                bar_postfix += [
                    f"\033[36mSubmap:{submap_id}\033[0m",
                    f"\033[36mFocal:{focal:.1f}",
                    f"\033[36mLocalKeyframes:{local_n_keyframes}\033[0m",
                    f"\033[36mGlobalKeyframes:{len(scene_model.keyframes)}\033[0m",
                ]
                pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

            pbar.update(1)
            prof.end_frame()

        if getattr(args, "online_pose_export_dir", ""):
            for keyframe in local_scene_model.keyframes:
                keyframe.info["online_Rt"] = (
                    keyframe.get_Rt().detach().cpu().clone()
                )
                keyframe.info["online_focal"] = (
                    torch.as_tensor(keyframe.f).detach().cpu().clone()
                )

        try:
            with prof.stage("final_ba.total"):
                final_ba_stats = run_anchored_ba(
                    pose_initializer,
                    local_scene_model,
                    local_scene_model.keyframes,
                    args,
                    f"Anchored submap BA {submap_id}",
                    return_point_colors=pose_debugger is not None,
                )
        except Exception:
            # Keep profiler diagnostics truthful for strict graph rejections and
            # backend exceptions, not only for solvers that return unusable.
            prof.incr_counter("final_ba_failures")
            raise
        if final_ba_stats is None or not final_ba_stats.get("solver_usable", False):
            prof.incr_counter("final_ba_failures")
            raise RuntimeError(
                f"Anchored submap BA {submap_id} did not produce a usable "
                "solution; refusing to export rollback poses as a successful final BA."
            )
        prof.incr_counter("final_ba_completed")
        if (
            args.final_submap_ba_optimize_focal
        ):
            focal = final_ba_stats["focal"]
            f = update_scene_focal(
                local_scene_model, local_scene_model.keyframes, focal
            )
        if final_ba_stats is not None and final_ba_stats.get("solver_usable", False):
            for keyframe in local_scene_model.keyframes:
                keyframe.desc_kpts.clear_3D_pts()
            for keyframe in local_scene_model.keyframes:
                keyframe.update_3dpts(local_scene_model.keyframes)
            if pose_debugger is not None:
                ba_points = final_ba_stats.get("points3d")
                if final_ba_stats.get("points3d_rgb") is not None:
                    ba_points = {
                        "xyz": final_ba_stats["points3d"],
                        "rgb": final_ba_stats["points3d_rgb"],
                        "groups": final_ba_stats.get("track_groups"),
                    }
                pose_debugger.update_ba(
                    local_scene_model.keyframes,
                    f,
                    ba_points,
                    label=f"submap {submap_id}",
                )

        append_submap_keyframes(scene_model, local_scene_model.keyframes, f)
        if final_ba_stats is not None and final_ba_stats.get("colmap_payload") is not None:
            final_ba_payloads.append(final_ba_stats["colmap_payload"])
        if pose_debugger is not None:
            pose_debugger.update_ba(
                scene_model.keyframes,
                f,
                "keep",
                label=f"global after submap {submap_id}",
            )

        # Release GPU memory accumulated by this submap before starting the next one.
        # In pose_debug_only mode the keyframes will not be touched on GPU again, so
        # also offload their dense per-frame tensors to CPU.
        release_submap_resources(
            local_scene_model,
            drop_dense=bool(getattr(args, "pose_debug_only", False)),
            label=f"submap {submap_id}",
            verbose=True,
        )
        del local_scene_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        prof.end_submap()
        if getattr(prof, "enabled", False):
            print(prof.format_brief())

    pbar.close()
    return scene_model, focal, f, final_ba_payloads


