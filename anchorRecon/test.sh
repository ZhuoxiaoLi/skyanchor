# ==============================================
# # ===== ablation: Original OTF-NVS / XFeat ====
# # ==============================================
# Run all scenes with:
#   RUN_ORIGINAL_OTF=1 bash test.sh
# Run one scene with:
#   RUN_ORIGINAL_OTF=1 ORIGINAL_OTF_SCENE=staff_ag bash test.sh
#
# This baseline intentionally removes all SkyAnchor-specific pose constraints:
# no --anchors_path, no --pose_ba_then_reconstruct, no --final_submap_ba,
# and no LoMa/LoMaG matcher arguments. The pose debug viewer is kept enabled
# so you can inspect the recovered trajectory in viser.
#
# Note: do not pass --init_cx/--init_cy here. Current train.py only supports
# custom principal points in the pose-debug path, while this baseline is the
# normal original OTF-NVS reconstruction path.
# if [[ "${RUN_ORIGINAL_OTF:-0}" == "1" ]]; then
#   set -e

#   run_original_otf_staff_ag() {
#   python train.py \
#     -s /data/albedo/skyanchor/on-the-fly-nvs/data/staff \
#     -m output/ablations/original_otf_xfeat/staff_ag \
#     --viewer_mode local 
 
    
#   }

#   run_original_otf_zeche() {
  # python train.py \
  #   -s /data/albedo/skyanchor/on-the-fly-nvs/data/zeche/ground \
  #   -m output/ablations/original_otf_xfeat/zeche \
  #   --viewer_mode local 
    
#   }

#   run_original_otf_park() {
#   python train.py \
#     -s /data/albedo/skyanchor/on-the-fly-nvs/data/park/ground \
#     -m output/ablations/original_otf_xfeat/park \
#     --viewer_mode local
    
#   }

#   run_original_otf_uni100k() {
#   # uni100k has reference poses under sparse-GT / recon-output instead of
#   # source_path/sparse/0, so train.py will not compute --eval_poses directly.
#   # Compare output/ablations/original_otf_xfeat/uni100k/colmap with evo.
#   python train.py \
#     -s /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground \
#     -m output/ablations/original_otf_xfeat/uni100k \
#     --viewer_mode local 
   
#   }

#   case "${ORIGINAL_OTF_SCENE:-all}" in
#     staff_ag)
#       run_original_otf_staff_ag
#       ;;
#     zeche)
#       run_original_otf_zeche
#       ;;
#     park)
#       run_original_otf_park
#       ;;
#     uni100k)
#       run_original_otf_uni100k
#       ;;
#     all)
#       run_original_otf_staff_ag
#       run_original_otf_zeche
#       run_original_otf_park
#       run_original_otf_uni100k
#       ;;
#     *)
#       echo "Unknown ORIGINAL_OTF_SCENE='${ORIGINAL_OTF_SCENE}'. Use staff_ag, zeche, park, uni100k, or all." >&2
#       exit 2
#       ;;
#   esac

#   exit 0
# fi


# ==============================================
# ===============ISPRS——std======================
# ==============================================

python train.py \
  -s /data/albedo/skyanchor/on-the-fly-nvs/data/isprs_std/ground \
  -m output/std/ground \
  --viewer_mode none \
  --pose_debug_viewer \
  --pose_debug_port 8090 \
  --random_seed 0 \
  --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/isprs_std/ground/submap_anchors.txt \
  --anchor_front 6 \
  --anchor_back 6 \
  --bootstrap_seed_init_mode anchor_pnp \
  --matcher_backend lomag \
  --loma_num_keypoints 4500 \
  --loma_magsac_threshold 1 \
  --loma_magsac_confidence 0.999999 \
  --loma_magsac_max_iters 50000 \
  --loma_filter_threshold 0.01 \
  --debug_pnp_matching \
  --pose_debug_only \
  --pose_debug_all_frames \
  --match_max_error 0.001 \
  --min_num_inliers 30 \
  --num_keyframes_miniba_bootstrap 8 \
  --num_pts_miniba_bootstrap 6000 \
  --iters_miniba_bootstrap 350 \
  --num_prev_keyframes_miniba_incr 4 \
  --num_prev_keyframes_miniba_incr_retry 6 \
  --num_prev_keyframes_check 6 \
  --keyframe_selection_use_3d \
  --pnpransac_samples 8000 \
  --num_pts_miniba_incr 6000 \
  --iters_miniba_incr 300 \
  --online_anchor_ba_every 10 \
  --online_anchor_ba_num_pts 6000 \
  --online_anchor_ba_iters 100 \
  --online_anchor_ba_candidate_multiplier 2 \
  --final_submap_ba_num_pts 10000 \
  --final_submap_ba_iters 400 \
  --lock_seed_anchor_poses \
  --final_submap_ba \
  --final_submap_ba_backend pycolmap \
  --final_submap_ba_pycolmap_solver dense-schur \
  --pose_ba_then_reconstruct \
  --final_submap_ba_candidate_multiplier 3 \
  --final_submap_ba_min_track_obs 3 \
  --final_submap_ba_anchor_match_frames 8 \
  --fix_focal \
  --init_focal 506.03 \
  --downsampling 2 \
  --final_submap_ba_max_depth 100.0 \
  --scene_scale 1 \
  --final_submap_ba_max_depth -1 \
  --online_pose_export_dir output/std/online_colmap \
  --final_submap_ba_export_dir output/std/ground/final_ba_colmap \
  --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/isprs_std/ground/sparse/0 \
  --pose_debug_ba_point_size 0.05 \
  --pose_debug_gt_point_size 0.05 \
  --pose_debug_frustum_scale 0.4 \
  --pose_debug_ba_frustum_scale 0.6 \
  --pose_debug_gt_frustum_scale 0.6  \
  --profile_timing \
  --profile_timing_dir output/std/ground/profiling \

# python train.py \
#   -s /data/albedo/skyanchor/on-the-fly-nvs/data/isprs_std/new-ground \
#   -m output/std/new_ground \
#   --viewer_mode none \
#   --pose_debug_viewer \
#   --pose_debug_port 8090 \
#   --random_seed 0 \
#   --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/isprs_std/new-ground/rec/anchorpose_11_00094_00193.txt \
#   --anchor_front 6 \
#   --anchor_back 6 \
#   --bootstrap_seed_init_mode anchor_pnp \
#   --matcher_backend lomag \
#   --loma_num_keypoints 4500 \
#   --loma_magsac_threshold 1 \
#   --loma_magsac_confidence 0.999999 \
#   --loma_magsac_max_iters 50000 \
#   --loma_filter_threshold 0.01 \
#   --debug_pnp_matching \
#   --pose_debug_only \
#   --pose_debug_all_frames \
#   --match_max_error 0.001 \
#   --min_num_inliers 30 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 4 \
#   --num_prev_keyframes_miniba_incr_retry 6 \
#   --num_prev_keyframes_check 6 \
#   --keyframe_selection_use_3d \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 300 \
#   --online_anchor_ba_every 10 \
#   --online_anchor_ba_num_pts 6000 \
#   --online_anchor_ba_iters 100 \
#   --online_anchor_ba_candidate_multiplier 2 \
#   --final_submap_ba_num_pts 10000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_min_track_obs 3 \
#   --final_submap_ba_anchor_match_frames 8 \
#   --fix_focal \
#   --init_focal 506.03 \
#   --downsampling 2 \
#   --final_submap_ba_max_depth 100.0 \
#   --scene_scale 1 \
#   --final_submap_ba_max_depth -1 \
#   --online_pose_export_dir output/std/new_ground/online_colmap \
#   --final_submap_ba_export_dir output/std/new_ground/final_ba_colmap \
#   --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/isprs_std/ground/sparse/0 \
#   --pose_debug_ba_point_size 0.05 \
#   --pose_debug_gt_point_size 0.05 \
#   --pose_debug_frustum_scale 0.4 \
#   --pose_debug_ba_frustum_scale 0.6 \
#   --pose_debug_gt_frustum_scale 0.6  \
#   --profile_timing \
#   --profile_timing_dir output/std/new_ground/profiling \


# ==============================================
# # ===============uni100k======================
# # ==============================================
python train.py \
  -s /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground \
  -m output/uni100k/ground \
  --viewer_mode none \
  --pose_debug_viewer \
  --pose_debug_port 8090 \
  --random_seed 0 \
  --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground/submap_anchors.txt \
  --anchor_front 6 \
  --anchor_back 6 \
  --bootstrap_seed_init_mode anchor_pnp \
  --matcher_backend lomag \
  --loma_num_keypoints 4500 \
  --loma_magsac_threshold 1 \
  --loma_magsac_confidence 0.999999 \
  --loma_magsac_max_iters 50000 \
  --loma_filter_threshold 0.01 \
  --debug_pnp_matching \
  --pose_debug_only \
  --pose_debug_all_frames \
  --match_max_error 0.001 \
  --min_num_inliers 30 \
  --num_keyframes_miniba_bootstrap 8 \
  --num_pts_miniba_bootstrap 6000 \
  --iters_miniba_bootstrap 350 \
  --num_prev_keyframes_miniba_incr 4 \
  --num_prev_keyframes_miniba_incr_retry 6 \
  --num_prev_keyframes_check 6 \
  --keyframe_selection_use_3d \
  --pnpransac_samples 8000 \
  --num_pts_miniba_incr 6000 \
  --iters_miniba_incr 300 \
  --online_anchor_ba_every 10 \
  --online_anchor_ba_num_pts 6000 \
  --online_anchor_ba_iters 100 \
  --online_anchor_ba_candidate_multiplier 2 \
  --final_submap_ba_num_pts 10000 \
  --final_submap_ba_iters 400 \
  --lock_seed_anchor_poses \
  --final_submap_ba \
  --final_submap_ba_backend pycolmap \
  --final_submap_ba_pycolmap_solver dense-schur \
  --pose_ba_then_reconstruct \
  --final_submap_ba_candidate_multiplier 3 \
  --final_submap_ba_min_track_obs 3 \
  --final_submap_ba_anchor_match_frames 8 \
  --fix_focal \
  --init_focal 674.1623393 \
  --downsampling 4 \
  --final_submap_ba_max_depth 100.0 \
  --scene_scale 10 \
  --final_submap_ba_max_depth -1 \
  --online_pose_export_dir output/uni100k/online_colmap \
  --final_submap_ba_export_dir output/uni100k/ground/final_ba_colmap \
  --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground/recon-output/rec \
  --pose_debug_ba_point_size 0.05 \
  --pose_debug_gt_point_size 0.05 \
  --pose_debug_frustum_scale 0.4 \
  --pose_debug_ba_frustum_scale 0.6 \
  --pose_debug_gt_frustum_scale 0.6  \
  --profile_timing \
  --profile_timing_dir output/uni100k/ground/profiling \


# python train.py \
#   -s /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground \
#   -m output/uni100k/ground \
#   --viewer_mode none \
#   --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground/submap_anchors.txt \
#   --anchor_front 6 \
#   --anchor_back 6 \
#   --matcher_backend lomag \
#   --loma_num_keypoints 4000 \
#   --loma_magsac_threshold 0.2 \
#   --loma_magsac_confidence 0.99999 \
#   --loma_magsac_max_iters 5000 \
#   --loma_filter_threshold 0.15 \
#   --debug_pnp_matching \
#   --pose_debug_viewer \
#   --pose_debug_only \
#   --pose_debug_port 8090 \
#   --match_max_error 0.001 \
#   --min_num_inliers 20 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 4 \
#   --num_prev_keyframes_check 10 \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 300 \
#   --final_submap_ba_num_pts 10000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_min_track_obs 3 \
#   --fix_focal \
#   --init_focal 2696.649357347645 \
#   --downsampling 1 \
#   --final_submap_ba_max_depth 500.0 \
#   --scene_scale 10 \
#   --final_submap_ba_export_dir output/uni100k/ground/final_ba_colmap \
#   --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/uni100k/ground/recon-output/rec \
#   --pose_debug_ba_point_size 0.05 \
#   --pose_debug_gt_point_size 0.03 \
#   --pose_debug_frustum_scale 0.1 \
#   --pose_debug_ba_frustum_scale 0.1 \
#   --pose_debug_gt_frustum_scale 0.1 \
#   --profile_timing \
#   --profile_timing_dir output/uni100k/ground/profiling \
  # --skip_incremental_magsac

# ==============================================
# ===============staff_ag======================
# ==============================================
set -e

# python train.py \
#   -s data/staff/ \
#   -m output/staff \
#   --viewer_mode none \
#   --pose_debug_viewer \
#   --pose_debug_port 8090 \
#   --random_seed 0 \
#   --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/staff/sub_anchors.txt \
#   --anchor_front 6 \
#   --anchor_back 6 \
#   --bootstrap_seed_init_mode anchor_pnp \
#   --matcher_backend lomag \
#   --loma_num_keypoints 6000 \
#   --loma_magsac_threshold 1 \
#   --loma_magsac_confidence 0.999999 \
#   --loma_magsac_max_iters 50000 \
#   --loma_filter_threshold 0.01 \
#   --debug_pnp_matching \
#   --pose_debug_only \
#   --pose_debug_all_frames \
#   --match_max_error 0.001 \
#   --min_num_inliers 30 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 4 \
#   --num_prev_keyframes_miniba_incr_retry 6 \
#   --num_prev_keyframes_check 6 \
#   --keyframe_selection_use_3d \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 300 \
#   --online_anchor_ba_every 10 \
#   --online_anchor_ba_num_pts 6000 \
#   --online_anchor_ba_iters 100 \
#   --online_anchor_ba_candidate_multiplier 2 \
#   --final_submap_ba_num_pts 10000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_min_track_obs 3 \
#   --final_submap_ba_anchor_match_frames 8 \
#   --fix_focal \
#   --downsampling 4 \
#   --init_focal 363.9831515 \
#   --final_submap_ba_max_depth -1 \
#   --online_pose_export_dir output/staff/online_colmap \
#   --final_submap_ba_export_dir output/staff/final_ba_colmap \
#   --profile_timing \
#   --profile_timing_dir output/staff/profiling

# python -B scripts/evaluate_pose_submaps.py \
#   --images-dir data/staff/images \
#   --gt data/staff/gt/sparse/0 \
#   --est output/staff/online_colmap \
#   --anchors data/staff/sub_anchors.txt \
#   --anchor-front 6 \
#   --anchor-back 6 \
#   --scale-min-observations 0 \
#   --json output/staff/online_metrics.json \
#   --csv output/staff/online_metrics.csv

python -B scripts/evaluate_pose_submaps.py \
  --images-dir data/staff/images \
  --gt data/staff/gt/sparse/0 \
  --est output/staff/final_ba_colmap \
  --anchors data/staff/sub_anchors.txt \
  --anchor-front 6 \
  --anchor-back 6 \
  --json output/staff/final_metrics.json \
  --csv output/staff/final_metrics.csv


#==============================================
#===============zeche======================
#==============================================
# python train.py \
#   -s data/zeche/ground/ \
#   -m output/zeche/ \
#   --viewer_mode none \
#   --anchors_path data/zeche/ground/submap_anchors.txt \
#   --anchor_front 6 \
#   --anchor_back 6 \
#   --matcher_backend lomag \
#   --loma_num_keypoints 6000 \
#   --loma_magsac_threshold 0.2 \
#   --loma_magsac_confidence 0.999999 \
#   --loma_magsac_max_iters 50000 \
#   --loma_filter_threshold 0.15 \
#   --debug_pnp_matching \
#   --pose_debug_viewer \
#   --pose_debug_only \
#   --pose_debug_port 8090 \
#   --match_max_error 0.001 \
#   --min_num_inliers 100 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 4 \
#   --num_prev_keyframes_check 10 \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 300 \
#   --final_submap_ba_num_pts 10000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_min_track_obs 3 \
#   --fix_focal \
#   --init_focal 1015.9457824383498 \
#   --init_cx 740.25 \
#   --init_cy 489.25 \
#   --downsampling 4 \
#   --final_submap_ba_max_depth 40.0 \
#   --final_submap_ba_export_dir output/zeche/final_ba_colmap \
#   --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/zeche/rec \
#   --pose_debug_ba_point_size 0.03 \
#   --pose_debug_gt_point_size 0.05 \
#   --pose_debug_frustum_scale 0.06 \
#   --pose_debug_ba_frustum_scale 0.2 \
#   --pose_debug_gt_frustum_scale 0.2 \


#==============================================
#===============park======================
#==============================================
# python train.py \
#   -s data/park/ground \
#   -m output/park/ \
#   --viewer_mode none \
#   --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/park/ground/submap_anchors.txt \
#   --anchor_front 6 \
#   --anchor_back 6 \
#   --matcher_backend lomag \
#   --loma_num_keypoints 6000 \
#   --loma_magsac_threshold 0.2 \
#   --loma_magsac_confidence 0.999999 \
#   --loma_magsac_max_iters 50000 \
#   --loma_filter_threshold 0.15 \
#   --debug_pnp_matching \
#   --pose_debug_viewer \
#   --pose_debug_only \
#   --pose_debug_port 8090 \
#   --match_max_error 0.001 \
#   --min_num_inliers 50 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 4 \
#   --num_prev_keyframes_check 10 \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 300 \
#   --final_submap_ba_num_pts 10000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_min_track_obs 3 \
#   --fix_focal \
#   --init_focal 618.1503889369058 \
#   --init_cx 790 \
#   --init_cy 447 \
#   --downsampling 1 \
#   --final_submap_ba_max_depth 100.0 \
#   --final_submap_ba_export_dir output/park/final_ba_colmap \
#   --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/park/sparse-all_output/rec \
#   --pose_debug_ba_point_size 0.03 \
#   --pose_debug_gt_point_size 0.05 \
#   --pose_debug_frustum_scale 0.06 \
#   --pose_debug_ba_frustum_scale 0.2 \
#   --pose_debug_gt_frustum_scale 0.2 \
#   --profile_timing \
#   --profile_timing_dir output/uni100k/ground/profiling 
python train.py \
  -s data/park/ground \
  -m output/park/ \
  --viewer_mode none \
  --pose_debug_viewer \
  --pose_debug_port 8090 \
  --random_seed 0 \
  --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/park/ground/submap_anchors.txt \
  --anchor_front 6 \
  --anchor_back 6 \
  --bootstrap_seed_init_mode anchor_pnp \
  --matcher_backend lomag \
  --loma_num_keypoints 6000 \
  --loma_magsac_threshold 1 \
  --loma_magsac_confidence 0.999999 \
  --loma_magsac_max_iters 50000 \
  --loma_filter_threshold 0.01 \
  --debug_pnp_matching \
  --pose_debug_only \
  --pose_debug_all_frames \
  --match_max_error 0.001 \
  --min_num_inliers 30 \
  --num_keyframes_miniba_bootstrap 8 \
  --num_pts_miniba_bootstrap 6000 \
  --iters_miniba_bootstrap 350 \
  --num_prev_keyframes_miniba_incr 4 \
  --num_prev_keyframes_miniba_incr_retry 6 \
  --num_prev_keyframes_check 6 \
  --keyframe_selection_use_3d \
  --pnpransac_samples 8000 \
  --num_pts_miniba_incr 6000 \
  --iters_miniba_incr 300 \
  --online_anchor_ba_every 10 \
  --online_anchor_ba_num_pts 6000 \
  --online_anchor_ba_iters 100 \
  --online_anchor_ba_candidate_multiplier 2 \
  --final_submap_ba_num_pts 10000 \
  --final_submap_ba_iters 400 \
  --lock_seed_anchor_poses \
  --final_submap_ba \
  --final_submap_ba_backend pycolmap \
  --final_submap_ba_pycolmap_solver dense-schur \
  --pose_ba_then_reconstruct \
  --final_submap_ba_candidate_multiplier 3 \
  --final_submap_ba_min_track_obs 3 \
  --final_submap_ba_anchor_match_frames 8 \
  --fix_focal \
  --init_focal 618.1503889369058  \
  --init_cx 790 \
  --init_cy 447 \
  --final_submap_ba_max_depth -1 \
  --online_pose_export_dir output/park/online_colmap \
  --final_submap_ba_export_dir output/park/final_ba_colmap \
  --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/park/sparse-all_output/rec \
  --pose_debug_ba_point_size 0.05 \
  --pose_debug_gt_point_size 0.05 \
  --pose_debug_frustum_scale 0.4 \
  --pose_debug_ba_frustum_scale 0.6 \
  --pose_debug_gt_frustum_scale 0.6  \
  --profile_timing \
  --profile_timing_dir output/staff/profiling

# python -B scripts/evaluate_pose_submaps.py \
#   --images-dir data/staff/images \
#   --gt data/staff/gt/sparse/0 \
#   --est output/staff/online_colmap \
#   --anchors data/staff/sub_anchors.txt \
#   --anchor-front 6 \
#   --anchor-back 6 \
#   --scale-min-observations 0 \
#   --json output/staff/online_metrics.json \
#   --csv output/staff/online_metrics.csv

# python -B scripts/evaluate_pose_submaps.py \
#   --images-dir data/staff/images \
#   --gt data/staff/gt/sparse/0 \
#   --est output/staff/final_ba_colmap \
#   --anchors data/staff/sub_anchors.txt \
#   --anchor-front 6 \
#   --anchor-back 6 \
#   --json output/staff/final_metrics.json \
#   --csv output/staff/final_metrics.csv

#==============================================
#===============isprs-rataus======================
#==============================================
# python train.py \
#   -s /data/albedo/skyanchor/on-the-fly-nvs/data/rat/ground \
#   -m /data/albedo/skyanchor/on-the-fly-nvs/data/rat/ground \
#   --viewer_mode none \
#   --anchors_path /data/albedo/skyanchor/on-the-fly-nvs/data/rat/ground/submap_anchors.txt \
#   --anchor_front 6 \
#   --anchor_back 6 \
#   --matcher_backend lomag \
#   --loma_num_keypoints 6000 \
#   --loma_magsac_threshold 0.2 \
#   --loma_magsac_confidence 0.999999 \
#   --loma_magsac_max_iters 50000 \
#   --loma_filter_threshold 0.15 \
#   --debug_pnp_matching \
#   --pose_debug_viewer \
#   --pose_debug_only \
#   --pose_debug_port 8090 \
#   --match_max_error 0.001 \
#   --min_num_inliers 50 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 4 \
#   --num_prev_keyframes_check 10 \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 300 \
#   --final_submap_ba_num_pts 10000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_min_track_obs 3 \
#   --fix_focal \
#   --init_focal 1016.813524236674 \
#   --init_cx 743.5 \
#   --init_cy 495.5 \
#   --downsampling 1 \
#   --final_submap_ba_max_depth 100.0 \
#   --final_submap_ba_export_dir /data/albedo/skyanchor/on-the-fly-nvs/data/rat/ground/final_ba_colmap \
#   --pose_debug_gt_colmap_dir /data/albedo/skyanchor/on-the-fly-nvs/data/rat/ground/rec \
#   --pose_debug_ba_point_size 0.03 \
#   --pose_debug_gt_point_size 0.05 \
#   --pose_debug_frustum_scale 0.06 \
#   --pose_debug_ba_frustum_scale 0.2 \
#   --pose_debug_gt_frustum_scale 0.2 \


# python train.py \
#   -s data/staff_ag/ \
#   -m output/test1_lomag_pose_dbg_strict1 \
#   --viewer_mode none \
#   --anchors_path data/staff_ag/submap_anchors.txt \
#   --anchor_front 4 \
#   --anchor_back 4 \
#   --matcher_backend lomag \
#   --loma_num_keypoints 10000 \
#   --loma_magsac_threshold 0.1 \
#   --loma_magsac_confidence 0.99999 \
#   --loma_magsac_max_iters 10000 \
#   --debug_pnp_matching \
#   --pose_debug_viewer \
#   --pose_debug_only \
#   --pose_debug_port 8090 \
#   --match_max_error 0.001 \
#   --min_num_inliers 30 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 10 \
#   --num_prev_keyframes_check 40 \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 60 \
#   --final_submap_ba_num_pts 50000 \
#   --final_submap_ba_iters 400 \
#   --final_submap_ba_backend pycolmap \
#   --final_submap_ba_pycolmap_solver dense-schur \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --final_submap_ba_optimize_focal \
#     --init_focal 363.9518900343642 \
#     --init_cx 455  \
#     --init_cy 340 \
#     --downsampling 1


#   --fix_focal \
  
# python train.py \
#   -s data/staff_ag/ \
#   -m output/test1_lomag_pose_dbg_strict1 \
#   --viewer_mode none \
#   --anchors_path data/staff_ag/submap_anchors.txt \
#   --anchor_front 4 \
#   --anchor_back 4 \
#   --matcher_backend lomag \
#   --loma_num_keypoints 4096 \
#   --loma_magsac_threshold 0.5 \
#   --loma_magsac_confidence 0.999999 \
#   --loma_magsac_max_iters 50000 \
#   --debug_pnp_matching \
#   --pose_debug_viewer \
#   --pose_debug_only \
#   --pose_debug_port 8090 \
#   --match_max_error 0.001 \
#   --min_num_inliers 150 \
#   --num_keyframes_miniba_bootstrap 8 \
#   --num_pts_miniba_bootstrap 6000 \
#   --iters_miniba_bootstrap 350 \
#   --num_prev_keyframes_miniba_incr 10 \
#   --num_prev_keyframes_check 40 \
#   --pnpransac_samples 8000 \
#   --num_pts_miniba_incr 6000 \
#   --iters_miniba_incr 60 \
#   --final_submap_ba_num_pts 12000 \
#   --final_submap_ba_iters 400 \
#   --lock_seed_anchor_poses \
#   --final_submap_ba \
#   --pose_ba_then_reconstruct \
#   --final_submap_ba_candidate_multiplier 3 \
#   --fix_focal \
#   --init_focal 363.9518900343642 \
#   --init_cx 455 \
#   --init_cy 340 \
#   --downsampling 1
