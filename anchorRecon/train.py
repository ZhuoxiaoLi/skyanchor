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

"""CLI entry point for anchored pose estimation and sparse reconstruction."""

from args import get_args


def main():
    # Parse first: --help and invalid configurations need no CUDA/model imports.
    args = get_args()
    import atexit
    import os
    import signal
    import time
    import torch
    from dataloaders.image_dataset import ImageDataset
    from poses.feature_detector import Detector
    from poses.loma_backend import build_loma_model, is_loma_backend
    from poses.matcher import Matcher
    from poses.pose_initializer import PoseInitializer
    from poses.triangulator import Triangulator
    from poses.pipeline import (
        seed_random_generators, run_anchor_submap_pose_pass,
        export_online_pose_sparse_model, export_final_ba_sparse_model,
    )
    from scene.pose_scene import PoseScene
    from utils import TimingProfiler, set_global_profiler
    if args.pose_debug_viewer:
        from poses.pose_debug_viewer import ViserPoseDebugger

    random_seed = seed_random_generators(args.random_seed)
    print(f"[reproducibility] random_seed={random_seed}")

    # Set up the global timing profiler so every module that calls
    # get_global_profiler() can record stages without us threading it through
    # every call site. When --profile_timing is not set, the singleton stays
    # at the no-op stub and every probe becomes a free fall-through.
    if getattr(args, "profile_timing", False):
        timing_profiler = TimingProfiler(
            enabled=True,
            sync_cuda=not getattr(args, "profile_timing_no_sync", False),
            track_per_frame=True,
        )
        set_global_profiler(timing_profiler)
        sync_msg = "cuda-sync ON" if timing_profiler.sync_cuda else "cuda-sync OFF"
        print(f"[profiler] enabled (per-frame breakdown, {sync_msg})")

        # Resolve where the partial / final report should be flushed. Done
        # here (before any heavy initialization that may segfault) so that
        # both atexit and SIGINT handlers always write to the same place.
        _profile_dir = (
            args.profile_timing_dir
            if args.profile_timing_dir
            else os.path.join(args.model_path, "profiling")
        )
        _profile_reported = {"done": False}

        def _emit_profile_report(reason: str = "atexit"):
            if _profile_reported["done"]:
                return
            _profile_reported["done"] = True
            try:
                timing_profiler.flush_pending()
                if not timing_profiler.frame_records:
                    print(f"[profiler/{reason}] no frames recorded; nothing to report.")
                    return
                print(f"[profiler/{reason}] dumping report to {_profile_dir} ...")
                timing_profiler.report(
                    out_dir=_profile_dir,
                    label=f"anchor_submap_pose_pass ({reason})",
                )
            except Exception as exc:
                print(f"[profiler/{reason}] failed to emit report: {exc}")

        atexit.register(_emit_profile_report, "atexit")

        def _sigint_handler(signum, frame):
            print(
                f"\n[profiler] received SIGINT; flushing partial timing report "
                f"to {_profile_dir} before exit."
            )
            _emit_profile_report("sigint")
            # Restore the default handler and re-raise so the user can still
            # break out cleanly (e.g. while the viser viewer's busy loop runs).
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, _sigint_handler)
    else:
        timing_profiler = None

    dataset = ImageDataset(args)
    height, width = dataset.get_image_size()
    if not dataset.anchor_submap_ranges:
        raise ValueError("No anchored submap ranges are available; check --anchors_path.")
    print("Initializing modules and running just in time compilation, may take a while...")
    max_error = max(args.match_max_error * width, 1.5)
    loma_model = None
    if is_loma_backend(args.matcher_backend):
        loma_model = build_loma_model(
            args.matcher_backend,
            args.num_kpts,
            args.loma_filter_threshold,
            args.loma_compile,
        )
    matcher = Matcher(
        args.match_min_cossim,
        args.matcher_backend,
        loma_model,
        args.loma_filter_threshold,
        args.loma_magsac_threshold,
        args.loma_magsac_confidence,
        args.loma_magsac_max_iters,
    )
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )
    pnp_max_error = (
        float(args.pnp_max_error_px)
        if getattr(args, "pnp_max_error_px", -1.0) > 0
        else 2 * max_error
    )
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, pnp_max_error, args
    )
    focal = pose_initializer.f_init
    f = torch.tensor([focal], device="cuda").float()
    pose_debugger = None
    if args.pose_debug_viewer:
        pose_debugger = ViserPoseDebugger(width, height, focal, args)
        pose_initializer.pose_debugger = pose_debugger
        scene_scale = float(getattr(args, "scene_scale", 1.0))
        if scene_scale != 1.0 and args.pose_debug_gt_colmap_dir:
            gt_export_dir = (
                args.scene_scale_gt_export_dir
                if getattr(args, "scene_scale_gt_export_dir", "")
                else os.path.join(args.model_path, "gt_colmap_scaled")
            )
            try:
                pose_debugger.export_scaled_gt_colmap(gt_export_dir, ext=".txt")
            except Exception as exc:
                print(f"[pose_debug] Failed to export scaled GT COLMAP to {gt_export_dir}: {exc}")
    scene_model = PoseScene(width, height, args, matcher)
    detector = Detector(args.num_kpts, width, height, args.matcher_backend, loma_model)
    print(f"Starting anchored pose estimation for {args.source_path}")
    scene_model, focal, f, sparse_payloads = run_anchor_submap_pose_pass(
        dataset, width, height, args, matcher, pose_initializer,
        triangulator, detector, scene_model, focal, pose_debugger,
    )
    export_online_pose_sparse_model(scene_model, args)
    export_final_ba_sparse_model(scene_model, sparse_payloads, args)
    if timing_profiler is not None:
        _emit_profile_report("post_pose_pass")
    if args.pose_debug_exit_after_export or pose_debugger is None:
        print("Pose pass finished; COLMAP exports saved.")
        return
    print("Pose pass finished. Keeping the viser viewer alive (Ctrl+C to exit).")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
