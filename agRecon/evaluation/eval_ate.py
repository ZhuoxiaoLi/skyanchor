import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pycolmap  # <-- NEW

import eval_util as tutil     # unchanged
from typing import Dict, Tuple

_logger = logging.getLogger(__name__)

def remove_images(recon: Path, remove_namelist: list):
    name2id = {}
    recon_out = pycolmap.Reconstruction(recon)
    for imid, im in recon_out.images.items():
        name2id[im.name] = imid

    for name in remove_namelist:
        recon_out.deregister_image(name2id[name])

    return recon_out


def load_poses(model_dir: Path) -> Dict[str, np.ndarray]:
    rec = pycolmap.Reconstruction(model_dir)
    poses = {}
    for img in rec.images.values():
        name = img.name   # keep file name only
        pose = img.cam_from_world.inverse()  # pycolmap.Rigid3d # cam to world
        R = pose.rotation.matrix()  # 3×3
        t = pose.translation.reshape(3, 1)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t[:, 0]
        poses[name] = T
    return poses

def load_poses_temp(model_dir: Path) -> Dict[str, np.ndarray]:
    rec = pycolmap.Reconstruction(model_dir)
    poses = {}
    for img in rec.images.values():
        name = img.name   # keep file name only
        if name.endswith(".jpg"):
            name = "a2_r1_L2/" + name

        pose = img.cam_from_world.inverse()  # pycolmap.Rigid3d # cam to world
        R = pose.rotation.matrix()  # 3×3
        t = pose.translation.reshape(3, 1)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t[:, 0]
        poses[name] = T
    return poses

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Compute pose errors (ATE / rotation) between two COLMAP "
                    "reconstructions using pycolmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "pred_rec",
        type=Path,
        help="Path to predicted COLMAP reconstruction folder (contains cameras.bin、images.bin、points3D.bin)",
    )
    parser.add_argument(
        "gt_rec",
        type=Path,
        help="Path to ground-truth COLMAP reconstruction folder",
    )

    parser.add_argument("--estimate_alignment", type=bool, default=True)
    parser.add_argument("--estimate_alignment_scale", type=bool, default=True)
    parser.add_argument("--pose_error_thresh_t", type=float, default=0.05)
    parser.add_argument("--pose_error_thresh_r", type=float, default=5)

    opt = parser.parse_args()

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    _logger.info("Reading COLMAP reconstructions…")
    _logger.info("[INFO] Loading ground‑truth model…")
    gt_poses = load_poses(opt.gt_rec)
    _logger.info(f"       Loaded {len(gt_poses)} GT images")

    _logger.info("[INFO] Loading predicted model…")
    pred_poses = load_poses(opt.pred_rec)
    _logger.info(f"       Loaded {len(pred_poses)} predicted images")

    common_names = sorted(set(gt_poses.keys()) & set(pred_poses.keys()))
    if not common_names:
        raise RuntimeError("No common images between GT and prediction models!")
    _logger.info(f"Found {len(common_names)} common images for evaluation.")

    pred_list, gt_list = [], []
    pred_list_, gt_list_ = [], []

    for name in common_names:
        if str(Path(name).parent) in ["aerial_h", "aerial_q","aerial_x","aerial_y","aerial_z"]:
            pred_list_.append((pred_poses[name],1000.0))
            gt_list_.append(gt_poses[name])

        pred_pose = pred_poses[name]
        gt_pose   = gt_poses[name]
        pred_list.append((pred_pose, 1000.0))
        gt_list.append(gt_pose)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    if opt.estimate_alignment:
        correspondences = [
            tutil.TestEstimate(pose_est=p_pose,
                               pose_gt=g_pose,
                               image_file=None, focal_length=None)
            for (p_pose, conf), g_pose in zip(pred_list_, gt_list_)
        ]

        align_T, align_s = tutil.estimate_alignment(
            estimates=correspondences,
            estimate_scale=opt.estimate_alignment_scale,
            inlier_threshold_r=opt.pose_error_thresh_r,
            inlier_threshold_t=opt.pose_error_thresh_t,
        )

        if align_T is None:
            _logger.warning("Alignment failed—error metrics will be ∞.")
    else:
        align_T, align_s = np.eye(4), 1.0

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    r_errs, t_errs = [], []
    hits = 0
    bad_imgs = []

    for name, ((pred_pose, _), gt_pose) in zip(common_names, zip(pred_list, gt_list)):
        if align_T is not None:
            gt_pose = align_T @ gt_pose
            t_err = np.linalg.norm(gt_pose[:3, 3] - pred_pose[:3, 3]) / align_s * 100
            dR = pred_pose[:3, :3] @ gt_pose[:3, :3].T
            r_err = np.linalg.norm(cv2.Rodrigues(dR)[0]) * 180 / math.pi
        else:
            t_err, r_err = math.inf, math.inf

        _logger.info(f"{name}  {r_err:6.2f}°  {t_err:7.2f} cm  ")
        r_errs.append(r_err)
        t_errs.append(t_err)
        if r_err < opt.pose_error_thresh_r and t_err / 100 < opt.pose_error_thresh_t:
            hits += 1

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    total = len(t_errs)
    ate_cm = float(np.mean(t_errs))                 # ← ATE
    avg_r  = float(np.mean(r_errs))
    acc    = hits / total * 100

    _logger.info("=" * 55)
    _logger.info("Evaluation complete.")
    _logger.info(f"Accuracy (<{opt.pose_error_thresh_r}° & <{opt.pose_error_thresh_t*100:.1f} cm): {acc:.1f}%")
    _logger.info(f"Average Rotation Error : {avg_r:.2f}°")
    _logger.info(f"Average Translation Error (ATE): {ate_cm:.2f} cm")

    # rec_out = remove_images(opt.pred_rec, bad_imgs)
    # rec_out.write("/home/dsxy/Project/agMatching/data/hkust-gz-2-testsmall/new_aerial_colmap/output/temp_473_delete")

# /home/dsxy/Project/agMatching/data/hkust-gz-2-testsmall/new_aerial_colmap/output/sfm_outputs/rec
# /home/dsxy/Project/agMatching/data/hkust-gz-2-testsmall/new_aerial_colmap/output/temp
# /home/dsxy/Project/agMatching/data/hkust-gz-21-a2g-test/gt-aerial-gound/merge3/all/sparse/0
# /home/dsxy/Project/agMatching/data/hkust-gz-21/sparse/0


