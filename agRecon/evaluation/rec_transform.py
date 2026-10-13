from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pycolmap

from .eval_util_2 import TestEstimate, estimate_alignment, estimate_alignment_all, estimate_alignment_lm_robust


def extract_poses_from_rec(rec: pycolmap.Reconstruction) -> Dict[str, np.ndarray]:
    """Return camera-to-world poses from a pycolmap Reconstruction as a dictionary mapping image basenames to 4x4 NumPy matrices."""
    poses: Dict[str, np.ndarray] = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue

        name = img.name

        cam2world = img.cam_from_world.inverse()

        R = cam2world.rotation.matrix()      # (3, 3)
        t = cam2world.translation.reshape(3) # (3,)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        poses[name] = T

    return poses


def estimate_sim3_between_pose_dicts(
    pred_poses: Dict[str, np.ndarray],
    gt_poses: Dict[str, np.ndarray],
    *,
    inlier_threshold_t: float = 0.05,
    inlier_threshold_r: float = 5.0,
    min_confident: int = 5,
    estimate_scale: bool = True,
    ransac_iterations: int = 10000,
    refinement_max_hyp: int = 12,
    refinement_max_it: int = 8,
) -> Tuple[pycolmap.Sim3d, pycolmap.Sim3d, List[str]]:
    """Estimate Sim3 alignment between GT and predicted pose dictionaries. Return sim_gt_to_pred and its inverse sim_pred_to_gt."""
    common_names = sorted(set(gt_poses.keys()) & set(pred_poses.keys()))
    if len(common_names) < min_confident:
        # raise RuntimeError(
        #     f"Too few common images for robust alignment: "
        #     f"{len(common_names)} < {min_confident}"
        # )
        print(f"Too few common images for robust alignment: {len(common_names)} < {min_confident}")
        return None, None, []

    estimates: List[TestEstimate] = []
    for name in common_names:
        pose_est = pred_poses[name]
        pose_gt = gt_poses[name]
        estimates.append(
            TestEstimate(
                pose_est=pose_est,
                pose_gt=pose_gt,
                focal_length=None,
                image_file=name,
            )
        )

    # T_gt_to_pred, scale_gt_to_pred = estimate_alignment(
    #     estimates=estimates,
    #     min_cofident_estimates=min_confident,
    #     inlier_threshold_t=inlier_threshold_t,
    #     inlier_threshold_r=inlier_threshold_r,
    #     estimate_scale=estimate_scale,
    #     ransac_iterations=ransac_iterations,
    #     refinement_max_hyp=refinement_max_hyp,
    #     refinement_max_it=refinement_max_it,
    # )

    T_gt_to_pred, scale_gt_to_pred = estimate_alignment_lm_robust(
        estimates=estimates,
        min_cofident_estimates=min_confident,
        estimate_scale=estimate_scale,
    )

    # T_gt_to_pred, scale_gt_to_pred = estimate_alignment_all(
    #     estimates=estimates,
    #     estimate_scale=estimate_scale,
    # )

    if T_gt_to_pred is None:
        # raise RuntimeError("Alignment failed, got no valid hypothesis.")
        return None, None, []

    s = float(scale_gt_to_pred)
    if s == 0.0:
        raise RuntimeError("Estimated scale is zero.")

    R = T_gt_to_pred[:3, :3]
    t = T_gt_to_pred[:3, 3]

    sim_gt_to_pred = pycolmap.Sim3d(
        scale=s,
        rotation=pycolmap.Rotation3d(matrix=R),
        translation=t,
    )
    sim_pred_to_gt = sim_gt_to_pred.inverse()
    return sim_gt_to_pred, sim_pred_to_gt, common_names


def align_reconstruction(
    pred_rec: pycolmap.Reconstruction,
    gt_rec: pycolmap.Reconstruction,
    *,
    min_common_images: int = 4,
    inlier_threshold_t: float = 0.05,
    inlier_threshold_r: float = 5.0,
    estimate_scale: bool = True,
    ransac_iterations: int = 10000,
    refinement_max_hyp: int = 12,
    refinement_max_it: int = 8,
) -> Tuple[pycolmap.Sim3d, List[str]]:
    """Align pred_rec to gt_rec in place and return (pred_rec, sim_pred_to_gt). Require min_common_images shared views; pass confidence_threshold, inlier_threshold_t, inlier_threshold_r, and estimate_scale to tutil.estimate_alignment."""
    gt_poses = extract_poses_from_rec(gt_rec)
    pred_poses = extract_poses_from_rec(pred_rec)

    sim_gt_to_pred, sim_pred_to_gt, common_names = estimate_sim3_between_pose_dicts(
        pred_poses=pred_poses,
        gt_poses=gt_poses,
        inlier_threshold_t=inlier_threshold_t,
        inlier_threshold_r=inlier_threshold_r,
        min_confident=min_common_images,
        estimate_scale=estimate_scale,
        ransac_iterations=ransac_iterations,
        refinement_max_hyp=refinement_max_hyp,
        refinement_max_it=refinement_max_it,
    )

    #    pred_world_new = sim_pred_to_gt(pred_world_old)
    if sim_gt_to_pred is None:
        return None, []

    pred_rec.transform(sim_pred_to_gt)

    for imid, img in pred_rec.images.items():
        if img.name in common_names:
            img.cam_from_world = gt_rec.image(imid).cam_from_world

    return sim_pred_to_gt, common_names

def compute_pose_errors(
    gt_poses: Dict[str, np.ndarray],
    pred_poses: Dict[str, np.ndarray],
    names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute translation and rotation errors for shared image names in GT and predicted pose dictionaries."""
    t_errors = []
    r_errors = []

    for name in names:
        T_gt = gt_poses[name]   # 4x4, camera-from-world
        T_pr = pred_poses[name] # 4x4, camera-from-world

        R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
        R_pr, t_pr = T_pr[:3, :3], T_pr[:3, 3]

        C_gt = -R_gt.T @ t_gt
        C_pr = -R_pr.T @ t_pr

        t_err = np.linalg.norm(C_pr - C_gt)

        dR = R_pr @ R_gt.T
        cos_angle = (np.trace(dR) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        r_err = np.degrees(angle_rad)

        t_errors.append(t_err)
        r_errors.append(r_err)

    return np.asarray(t_errors), np.asarray(r_errors)
