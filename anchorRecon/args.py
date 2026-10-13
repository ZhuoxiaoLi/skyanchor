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

import argparse
import os

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Anchored pose estimation and sparse COLMAP reconstruction")

    parser.add_argument('--random_seed', type=int, default=0,
                        help="Random seed for PyTorch, NumPy, Python, and OpenCV sampling. Use this for reproducible pose/track ablations.")

    ## Data and Images
    parser.add_argument('-s', '--source_path', type=str, required=True,
                        help="Path to the data folder (should have sparse/0/ if using COLMAP or evaluating poses)")
    parser.add_argument('-i', '--images_dir', type=str, default="images",
                        help="source_path/images_dir is the path to the images (with extensions jpg, png or jpeg).")
    parser.add_argument('--masks_dir', type=str, default="", 
                        help="If set, source_path/masks_dir is the path to optional masks to apply to the images before computing loss (png).")
    parser.add_argument('--num_loader_threads', type=int, default=4,
                        help="Number of workers to load and prepare input images")
    parser.add_argument('--downsampling', type=float, default=-1.0, help="Downsampling ratio for input images")
    parser.add_argument('--min_displacement', type=float, default=0.03,
                        help="Minimum median keypoint displacement for a new keyframe to be added. Relative to the image width")
    parser.add_argument('--start_at', type=int, default=0,
                        help="Number of frames to skip from the dataset.")
    

    ## COLMAP options
    parser.add_argument('--eval_poses', action='store_true',
                        help="Compare poses to COLMAP")
    parser.add_argument('--use_colmap_poses', action='store_true',
                        help="Load COLMAP data for pose and intrinsics initialization")
    parser.add_argument('--anchors_path', type=str, required=True,
                        help="Path to an anchors.txt file with world-to-camera poses in the format: image_name tx ty tz qw qx qy qz")
    parser.add_argument('--scene_scale', type=float, default=1.0,
                        help="Multiplicative scale applied to imported anchor translations and to the GT COLMAP "
                             "model loaded via --pose_debug_gt_colmap_dir (poses + 3D points). Useful when the SfM "
                             "world units are too small (e.g. no RTK) and the triangulator's disambiguation threshold "
                             "(world-unit depth=1..10) collapses. Set e.g. 10 or 100 to enlarge the world. Rotations, "
                             "intrinsics, pixel-space thresholds, and frustum scales are NOT touched. If a GT COLMAP "
                             "directory is provided and scene_scale != 1, a rescaled copy is written to "
                             "<model_path>/gt_colmap_scaled (or --scene_scale_gt_export_dir).")
    parser.add_argument('--scene_scale_gt_export_dir', type=str, default="",
                        help="Override the output directory for the rescaled GT COLMAP model "
                             "(active only when --scene_scale != 1 and --pose_debug_gt_colmap_dir is set). "
                             "Defaults to <model_path>/gt_colmap_scaled.")
    parser.add_argument(
        '--num_seed_anchor_frames', '--anchor_front', type=int, default=4,
        help="Number of earliest anchor frames used for seeded bootstrap. These frames are expected to appear first in the current submap.")
    parser.add_argument(
        '--num_tail_anchor_frames', '--anchor_back', type=int, default=-1,
        help="Number of latest anchor frames used to close each anchored submap. Defaults to --num_seed_anchor_frames.")
    parser.add_argument(
        '--anchor_submap_size', '--submap_size', type=int, default=0,
        help="If > 0, split the input sequence into fixed-size anchored pose submaps. If 0, infer anchored submaps from --anchors_path using --anchor_front/--anchor_back when possible.")
    parser.add_argument('--lock_seed_anchor_poses', action='store_true',
                        help="Compatibility flag: seed anchor poses remain fixed during anchored BA.")
    parser.add_argument('--bootstrap_seed_init_mode',
                        choices=['anchor_pnp'],
                        default='anchor_pnp',
                        help="Initialize free bootstrap cameras with fixed-anchor triangulation and PnP, then jointly refine with MiniBA.")
    parser.add_argument('--final_submap_ba', action='store_true', default=True,
                        help="Run a final pose bundle adjustment at the end of reconstruction while keeping all anchor poses fixed.")
    parser.add_argument('--final_submap_ba_backend', type=lambda s: s.lower().replace("_", "-"), default="pycolmap",
                        choices=["pycolmap"],
                        help="Anchored BA uses pycolmap and pyceres in the active environment.")
    parser.add_argument('--final_submap_ba_pycolmap_solver', type=lambda s: s.lower().replace("_", "-"), default="dense-schur",
                        choices=["dense-schur", "iterative-schur", "sparse-schur", "dense-qr", "sparse-normal-cholesky"],
                        help="Linear solver used by the pycolmap final BA backend. Use dense-schur or iterative-schur to avoid MKL sparse solver issues.")
    parser.add_argument('--final_submap_ba_num_pts', type=int, default=4000,
                        help="Number of feature tracks used in the final anchored submap BA.")
    parser.add_argument('--final_submap_ba_candidate_multiplier', type=float, default=3.0,
                        help="Build this many more candidate tracks for final submap BA, then keep the tracks with the most anchor-connected observations.")
    parser.add_argument('--final_submap_ba_min_track_obs', type=int, default=3,
                        help="Minimum number of image observations preferred for final submap BA tracks. Lower-observation tracks are used only as fallback if needed.")
    parser.add_argument('--final_submap_ba_anchor_match_frames', type=int, default=8,
                        help="Before final BA, explicitly match every tail anchor to this many last movable frames so the back boundary forms a real loop closure. Set 0 to disable.")
    parser.add_argument('--final_submap_ba_max_depth', type=float, default=-1.0,
                        help="If > 0, keep only final BA tracks whose closest positive observed depth is at most this value.")
    parser.add_argument('--final_submap_ba_iters', type=int, default=100,
                        help="Number of Levenberg-Marquardt iterations used in the final anchored submap BA.")
    parser.add_argument('--final_submap_ba_optimize_focal', action='store_true',
                        help="Also refine the focal length during the final anchored submap BA.")
    parser.add_argument('--final_submap_ba_export_dir', type=str, default="",
                        help="Directory where the final BA sparse model is exported as COLMAP text. Defaults to <model_path>/final_ba_colmap when final BA is enabled.")
    parser.add_argument('--online_pose_export_dir', type=str, default="",
                        help="Optional directory where anchored-submap poses are exported as a points-free COLMAP text model immediately before final BA.")
    parser.add_argument('--online_anchor_ba_every', type=int, default=0,
                        help="If > 0, run a causal front-anchor-only BA after this many newly registered movable cameras. A final flush is also run immediately before the first tail anchor. Set 0 to disable.")
    parser.add_argument('--online_anchor_ba_num_pts', type=int, default=6000,
                        help="Number of feature tracks used by each causal front-anchor-only BA.")
    parser.add_argument('--online_anchor_ba_iters', type=int, default=100,
                        help="Maximum solver iterations for each causal front-anchor-only BA.")
    parser.add_argument('--online_anchor_ba_candidate_multiplier', type=float, default=2.0,
                        help="Candidate-track multiplier for causal front-anchor-only BA.")
    parser.add_argument('--pose_ba_then_reconstruct', action='store_true', default=True,
                        help="Compatibility flag: the pipeline always estimates poses and runs final anchored BA; no Gaussian reconstruction.")
        
        

    ## Pose initialization options
    # Matching
    parser.add_argument('--matcher_backend', type=lambda s: s.lower().replace("_", "-"), default="lomag",
                        choices=["lomag", "loma-g"],
                        help="Feature matching backend. Use lomag/loma-g to run LoMa-G keypoints, descriptors and matcher.")
    parser.add_argument('--num_kpts', type=int, default=int(4096*4),
                        help="Number of keypoints to extract from each image")
    parser.add_argument('--match_max_error', type=float, default=2e-3,
                        help="Maximum reprojection error for matching keypoints, proportion of the image width. This is used to filter outliers and discard points at triangulation.")
    parser.add_argument('--match_min_cossim', type=float, default=0.82,
                        help="Minimum cosine similarity for mutual-nearest-neighbor descriptor matches.")
    parser.add_argument('--loma_filter_threshold', type=float, default=-1.0,
                        help="LoMa match filter threshold. If negative, use the selected LoMa model default.")
    parser.add_argument('--loma_num_keypoints', type=int, default=2048,
                        help="Number of LoMa keypoints to detect per image. Used instead of --num_kpts for LoMa backends.")
    parser.add_argument('--loma_compile', action='store_true',
                        help="Compile the LoMa model on supported platforms.")
    parser.add_argument('--loma_magsac_threshold', type=float, default=0.2,
                        help="USAC_MAGSAC reprojection threshold in pixels for LoMa match filtering.")
    parser.add_argument('--loma_magsac_confidence', type=float, default=0.999999,
                        help="USAC_MAGSAC confidence for LoMa match filtering.")
    parser.add_argument('--loma_magsac_max_iters', type=int, default=10000,
                        help="USAC_MAGSAC max iterations for LoMa match filtering.")
    parser.add_argument('--min_num_inliers', type=int, default=100,
                        help="The keyframe will be added only if the number of inliers is greater than this value")
    parser.add_argument('--debug_pnp_matching', action='store_true',
                        help="Print detailed match and inlier counts when incremental PnP initialization fails.")
    parser.add_argument('--pose_debug_viewer', action='store_true',
                        help="Start a lightweight viser viewer for PnP matches, poses, BA poses, and BA 3D points.")
    parser.add_argument('--pose_debug_only', action='store_true', default=True,
                        help="Run only the pose/BA debug pass and keep the viser viewer alive. Intended with --pose_ba_then_reconstruct and --anchors_path.")
    parser.add_argument('--pose_debug_all_frames', action='store_true',
                        help="In the anchored pose pass, estimate every input frame instead of dropping low-displacement non-keyframes. This gives complete trajectory coverage for pose-only evaluation.")
    parser.add_argument('--pose_debug_no_viewer', action='store_true',
                        help="Run --pose_debug_only headlessly and exit immediately after exporting the sparse pose model.")
    parser.add_argument('--pose_debug_exit_after_export', action='store_true',
                        help="When used with --pose_debug_only, export the final BA COLMAP model and exit instead of keeping the viser viewer alive.")
    parser.add_argument('--pose_debug_host', type=str, default="0.0.0.0",
                        help="Host for the viser pose debug viewer.")
    parser.add_argument('--pose_debug_port', type=int, default=8090,
                        help="Port for the viser pose debug viewer.")
    parser.add_argument('--pose_debug_max_points', type=int, default=50000,
                        help="Maximum number of 3D points shown in the viser pose debug viewer.")
    parser.add_argument('--pose_debug_match_step', type=int, default=2,
                        help="Draw every Nth match in the viser match image panel.")
    parser.add_argument('--pose_debug_frustum_scale', type=float, default=0.06,
                        help="Camera frustum scale in the viser pose debug viewer.")
    parser.add_argument('--pose_debug_ba_frustum_scale', type=float, default=-1.0,
                        help="Camera frustum scale for BA poses in the viser pose debug viewer. If negative, reuse --pose_debug_frustum_scale.")
    parser.add_argument('--pose_debug_gt_frustum_scale', type=float, default=-1.0,
                        help="Camera frustum scale for GT poses in the viser pose debug viewer. If negative, reuse --pose_debug_frustum_scale.")
    parser.add_argument('--pose_debug_ba_point_size', type=float, default=0.012,
                        help="Base point size for BA-related points in the viser pose debug viewer.")
    parser.add_argument('--pose_debug_gt_point_size', type=float, default=0.01,
                        help="Point size for GT sparse points in the viser pose debug viewer.")
    parser.add_argument('--pose_debug_gt_colmap_dir', type=str, default="",
                        help="Optional COLMAP model directory to overlay as GT in the viser pose debug viewer.")
    # Initial mini bundle adjustment
    parser.add_argument('--num_keyframes_miniba_bootstrap', type=int, default=8,
                        help="Number of first keyframes accumulated for pose and focal estimation before optimization")
    parser.add_argument('--num_pts_miniba_bootstrap', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_bootstrap', type=int, default=200)
    # Focal estimation
    parser.add_argument('--fix_focal', action='store_true', 
                        help="If set, will use init_focal or init_fov without reoptimizing focal")
    parser.add_argument('--init_focal', type=float, default=-1.0, 
                        help="Initial focal length in pixels. If not set, will use init_fov or be set as 0.7*width of the image if init_fov is also not set")
    parser.add_argument('--init_fov', type=float, default=-1.0, 
                        help="Initial horizontal FoV in degrees. Used only if init_focal is not set")
    parser.add_argument('--init_cx', type=float, default=None,
                        help="Initial principal point x in pixels. If omitted, use the current image center convention ((width-1)/2).")
    parser.add_argument('--init_cy', type=float, default=None,
                        help="Initial principal point y in pixels. If omitted, use the current image center convention ((height-1)/2).")
    # Incremental pose optimization
    parser.add_argument('--num_prev_keyframes_miniba_incr', type=int, default=8,
                        help="Number of previous keyframes for incremental pose initialization")
    parser.add_argument('--num_prev_keyframes_miniba_incr_retry', type=int, default=0,
                        help="If greater than --num_prev_keyframes_miniba_incr, retry a failed incremental PnP with this many 3D-ranked reference keyframes. Set 0 to disable.")
    parser.add_argument('--skip_incremental_magsac', action='store_true',
                        help="Skip the MAGSAC fundamental-matrix outlier filter inside initialize_incremental "
                             "(the per-keyframe matcher loop). PnP-RANSAC will still filter 2D-3D outliers "
                             "via reprojection error, so this is mostly safe and saves "
                             "O(num_prev_keyframes_miniba_incr * 100-300ms) per keyframe addition. "
                             "Bootstrap and final-BA pairwise matching keep MAGSAC (different role).")
    parser.add_argument('--num_prev_keyframes_check', type=int, default=30,
                        help="Number of previous keyframes to check for matches with new keyframe")
    parser.add_argument('--keyframe_selection_use_3d', action='store_true',
                        help="When selecting previous keyframes for incremental PnP, rank candidates by "
                             "the number of cheap descriptor matches whose keypoints already have 3D points. "
                             "This avoids selecting recent keyframes that have many 2D matches but no usable "
                             "3D points, which can break the PnP chain in long submaps.")
    parser.add_argument('--pnp_max_depth', type=float, default=-1.0,
                        help="Incremental PnP always rejects non-finite and behind-reference-camera 3D points. If this value is > 0, also reject points whose reference-camera depth exceeds it; useful for far, low-parallax geometry.")
    parser.add_argument('--pnp_max_error_px', type=float, default=-1.0,
                        help="If > 0, use this pixel reprojection threshold for incremental PnP-RANSAC. "
                             "By default, the threshold is 2 * max(match_max_error * image_width, 1.5).")
    parser.add_argument('--pnpransac_samples', type=int, default=1000,
                        help="Maximum number of set of 2D-3D matches used to estimate the initial pose and outlier removal")
    parser.add_argument('--num_pts_miniba_incr', type=int, default=2000,
                        help="Number of keypoints considered for initial mini bundle adjustment")
    parser.add_argument('--iters_miniba_incr', type=int, default=20)




    ## Evaluation
    parser.add_argument('--test_hold', type=int, default=-1, 
                        help="Mark every Nth image as a test frame for pose reference selection; -1 disables test-frame marking.")
    parser.add_argument('--profile_timing', action='store_true',
                        help="Enable the hierarchical timing profiler over the anchored pose pass. "
                             "Wraps detector / matcher / PnP / miniBA / final BA stages with CUDA-synchronized "
                             "timers and prints a summary table at the end.")
    parser.add_argument('--profile_timing_dir', type=str, default="",
                        help="If --profile_timing is set, write per-frame CSV and JSON summaries into this directory. "
                             "If empty, defaults to <model_path>/profiling.")
    parser.add_argument('--profile_timing_no_sync', action='store_true',
                        help="If --profile_timing is set, do NOT call torch.cuda.synchronize() around each stage. "
                             "Faster but timings may overlap with subsequent kernels. Off by default.")

    ## Checkpoint options
    parser.add_argument('-m', '--model_path', default="", 
                        help="Base directory for sparse COLMAP exports and profiling. Defaults to results/xxxxxx.")

    ## Viewer
    parser.add_argument('--viewer_mode', choices=['none'], default='none')

    args = parser.parse_args(argv)
    if "://" in args.source_path:
        parser.error("--source_path must be a local image dataset, not a stream URL.")
    if not args.anchors_path:
        parser.error("--anchors_path must not be empty.")
    if args.num_seed_anchor_frames < 2:
        parser.error("Anchored PnP bootstrap requires at least two seed anchors.")
    if args.num_keyframes_miniba_bootstrap < args.num_seed_anchor_frames:
        parser.error("--num_keyframes_miniba_bootstrap must be at least --anchor_front.")
    if args.pose_debug_only and not args.pose_debug_no_viewer:
        args.pose_debug_viewer = True
    if args.pose_debug_all_frames and not args.pose_debug_only:
        parser.error("--pose_debug_all_frames requires --pose_debug_only.")
    if args.pose_debug_no_viewer:
        args.pose_debug_viewer = False
        args.pose_debug_exit_after_export = True
    if args.num_tail_anchor_frames < 0:
        args.num_tail_anchor_frames = args.num_seed_anchor_frames
    if (args.init_cx is None) != (args.init_cy is None):
        parser.error("--init_cx and --init_cy must be provided together.")
    if args.loma_num_keypoints <= 0:
        parser.error("--loma_num_keypoints must be positive.")
    args.num_kpts = args.loma_num_keypoints
    if args.anchor_submap_size < 0:
        parser.error("--anchor_submap_size must be non-negative.")
    if args.num_seed_anchor_frames < 0:
        parser.error("--num_seed_anchor_frames must be non-negative.")
    if args.num_tail_anchor_frames < 0:
        parser.error("--num_tail_anchor_frames must be non-negative.")
    if args.num_prev_keyframes_miniba_incr < 1:
        parser.error("--num_prev_keyframes_miniba_incr must be positive.")
    if args.num_prev_keyframes_check < 1:
        parser.error("--num_prev_keyframes_check must be positive.")
    if args.num_prev_keyframes_miniba_incr > args.num_prev_keyframes_check:
        parser.error(
            "--num_prev_keyframes_miniba_incr cannot exceed "
            "--num_prev_keyframes_check."
        )
    if args.num_prev_keyframes_miniba_incr_retry < 0:
        parser.error("--num_prev_keyframes_miniba_incr_retry must be non-negative.")
    if args.num_prev_keyframes_miniba_incr_retry:
        if (
            args.num_prev_keyframes_miniba_incr_retry
            <= args.num_prev_keyframes_miniba_incr
        ):
            parser.error(
                "A non-zero --num_prev_keyframes_miniba_incr_retry must exceed "
                "--num_prev_keyframes_miniba_incr."
            )
        if (
            args.num_prev_keyframes_miniba_incr_retry
            > args.num_prev_keyframes_check
        ):
            parser.error(
                "--num_prev_keyframes_miniba_incr_retry cannot exceed "
                "--num_prev_keyframes_check."
            )
    if args.pnpransac_samples <= 0:
        parser.error("--pnpransac_samples must be positive.")
    if args.min_num_inliers < 4:
        parser.error("--min_num_inliers must be at least 4 for P4P-RANSAC.")
    if args.num_pts_miniba_incr <= 0:
        parser.error("--num_pts_miniba_incr must be positive.")
    if args.min_num_inliers > args.num_pts_miniba_incr:
        parser.error(
            "--min_num_inliers cannot exceed --num_pts_miniba_incr; "
            "incremental MiniBA could never accept a pose."
        )
    if args.iters_miniba_incr <= 0:
        parser.error("--iters_miniba_incr must be positive.")
    if args.final_submap_ba_num_pts <= 0:
        parser.error("--final_submap_ba_num_pts must be positive.")
    if args.final_submap_ba_iters <= 0:
        parser.error("--final_submap_ba_iters must be positive.")
    if args.final_submap_ba_candidate_multiplier < 1.0:
        parser.error("--final_submap_ba_candidate_multiplier must be at least 1.")
    if args.final_submap_ba_min_track_obs < 2:
        parser.error("--final_submap_ba_min_track_obs must be at least 2.")
    if args.final_submap_ba_anchor_match_frames < 0:
        parser.error("--final_submap_ba_anchor_match_frames must be non-negative.")
    if args.online_anchor_ba_every < 0:
        parser.error("--online_anchor_ba_every must be non-negative.")
    if args.online_anchor_ba_num_pts <= 0:
        parser.error("--online_anchor_ba_num_pts must be positive.")
    if args.online_anchor_ba_iters <= 0:
        parser.error("--online_anchor_ba_iters must be positive.")
    if args.online_anchor_ba_candidate_multiplier < 1.0:
        parser.error("--online_anchor_ba_candidate_multiplier must be at least 1.")
    if args.online_anchor_ba_num_pts < args.min_num_inliers:
        parser.error(
            "--online_anchor_ba_num_pts cannot be smaller than "
            "--min_num_inliers."
        )

    ## Set the output directory if not specified
    if args.model_path == "":
        i = 0
        while os.path.exists(f"results/{i:06d}"):
            i += 1
        args.model_path = f"results/{i:06d}"

    return args
