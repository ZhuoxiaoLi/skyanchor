#!/usr/bin/env python3
"""
Pose Error Evaluator
====================

Given two COLMAP reconstructions (ground‑truth and predicted) stored in the standard
binary/text format, compute per‑pair relative rotation error (RRE) and relative
translation direction error (RTE) and report accuracy at several thresholds.

The script uses **pycolmap** for IO, so no custom readers are required.

Usage
-----
python pose_error_evaluator.py \
    --gt_model_dir   /path/to/gt/model \
    --pred_model_dir /path/to/pred/model \
    --pairs_file     /optional/pairs.txt   # each line: img1 img2

If *--pairs_file* is omitted, the tool will evaluate **all** common image pairs.

Outputs
-------
* Printed table of RRA / RTA @ (5,10,15,30) degrees
* JSON file *pose_errors.json* containing the raw per‑pair errors
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pycolmap  # pip install pycolmap

############################
# Geometry helper routines #
############################

def rotation_matrix_from_qvec(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion [qw, qx, qy, qz] to 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ], dtype=float)


def rot_angle_deg(R_err: np.ndarray) -> float:
    """Smallest angle (deg) that rotates R_err to identity."""
    # Clamp for numerical stability
    cos_theta = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Angular error between two rotation matrices (deg)."""
    return rot_angle_deg(R_gt @ R_pred.T)


def translation_dir_error_deg(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """Angle (deg) between two translation **direction** vectors."""
    t_pred_n = t_pred / np.linalg.norm(t_pred)
    t_gt_n = t_gt / np.linalg.norm(t_gt)
    cos_theta = np.clip(t_pred_n.dot(t_gt_n), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

########################
# COLMAP data handling #
########################

def load_poses(model_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load camera poses from a COLMAP *model_dir* using pycolmap.

    Returns
    -------
    dict mapping image name → (R, t) where the pose is **world→camera** such that
    X_cam = R * X_world + t.
    """
    rec = pycolmap.Reconstruction(model_dir)
    poses = {}
    for img in rec.images.values():
        name = Path(img.name).name  # keep file name only
        pose = img.cam_from_world  # pycolmap.Rigid3d
        R = pose.rotation.matrix()  # 3×3
        t = pose.translation.reshape(3)
        poses[name] = (R, t)
    return poses

###############################
# Evaluation over image pairs #
###############################

def build_pair_list(names: List[str], pair_file: Path = None) -> List[Tuple[str, str]]:
    if pair_file is None:
        # Dense list of all ordered pairs i→j, i ≠ j
        pairs = []
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                pairs.append((n1, n2))
        return pairs
    pairs = []
    with pair_file.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            a, b = line.strip().split()
            pairs.append((Path(a).name, Path(b).name))
    return pairs

###################
# Main evaluation #
###################

def evaluate(gt_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
             pred_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
             pairs: List[Tuple[str, str]]):
    r_errors, t_errors = [], []

    for im1, im2 in pairs:
        if im1 not in gt_poses or im2 not in gt_poses:
            print(f"[WARNING] Pair ({im1},{im2}) missing in ground truth – skipped.")
            continue
        if im1 not in pred_poses or im2 not in pred_poses:
            print(f"[WARNING] Pair ({im1},{im2}) missing in prediction   – skipped.")
            continue

        # Relative transforms (world→camera)
        R1_gt, t1_gt = gt_poses[im1]
        R2_gt, t2_gt = gt_poses[im2]
        R1_pr, t1_pr = pred_poses[im1]
        R2_pr, t2_pr = pred_poses[im2]

        # Relatives: Cj_T_Ci = Tj * inv(Ti)
        R_rel_gt = R2_gt @ R1_gt.T
        t_rel_gt = t2_gt - R_rel_gt @ t1_gt  # standard formulation

        R_rel_pr = R2_pr @ R1_pr.T
        t_rel_pr = t2_pr - R_rel_pr @ t1_pr

        r_err = rotation_error_deg(R_rel_pr, R_rel_gt)
        t_err = translation_dir_error_deg(t_rel_pr, t_rel_gt)

        r_errors.append(r_err)
        t_errors.append(t_err)

    return np.array(r_errors), np.array(t_errors)

#########################
# Accuracy computations #
#########################

THRESHOLDS = [5, 10, 15, 30]

def accuracy(errors: np.ndarray, thr: float) -> float:
    if len(errors) == 0:
        return 0.0
    return 100.0 * np.mean(errors < thr)

################
# Entry point  #
################

def parse_args():
    p = argparse.ArgumentParser(description="Pose error evaluation using pycolmap")
    p.add_argument("--gt_model_dir",   type=Path, required=True, help="COLMAP GT model directory")
    p.add_argument("--pred_model_dir", type=Path, required=True, help="COLMAP predicted model directory")
    p.add_argument("--pairs_file", type=Path, help="Optional text file specifying image pairs")
    p.add_argument("--out", type=Path, default=None, help="JSON file to save raw errors")
    return p.parse_args()

def main():
    args = parse_args()

    print("[INFO] Loading ground‑truth model…")
    gt_poses = load_poses(args.gt_model_dir)
    print(f"       Loaded {len(gt_poses)} GT images")

    print("[INFO] Loading predicted model…")
    pred_poses = load_poses(args.pred_model_dir)
    print(f"       Loaded {len(pred_poses)} predicted images")

    common_names = sorted(set(gt_poses.keys()) & set(pred_poses.keys()))
    if not common_names:
        raise RuntimeError("No common images between GT and prediction models!")

    pairs = build_pair_list(common_names, args.pairs_file)
    print(f"[INFO] Evaluating {len(pairs)} image pairs\n")

    r_err, t_err = evaluate(gt_poses, pred_poses, pairs)

    # Print summary
    for thr in THRESHOLDS:
        print(f"RRA @{thr:2d}° : {accuracy(r_err, thr):6.2f} %")
    print("-" * 32)
    for thr in THRESHOLDS:
        print(f"RTA @{thr:2d}° : {accuracy(t_err, thr):6.2f} %")

    # Save raw errors
    if args.out:
        results = {
            "pair": [f"{a}|{b}" for a, b in pairs],
            "rError": r_err.tolist(),
            "tError": t_err.tolist(),
        }
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Saved raw errors to {args.out}")


if __name__ == "__main__":
    main()
