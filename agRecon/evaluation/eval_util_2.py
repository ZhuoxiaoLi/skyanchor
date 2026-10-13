import math
import random
from collections import namedtuple
from scipy.spatial.transform import Rotation
import numpy as np
import logging
import pycolmap

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

TestEstimate = namedtuple("TestEstimate", [
    "pose_est",
    "pose_gt",
    "focal_length",
    "image_file"
])


def colmap_cam2world(image: pycolmap.Image) -> np.ndarray:
    """Return 4×4 cam-to-world pose matrix from a pycolmap.Image object."""
    # pycolmap stores cam_from_world (world->cam). We want cam->world.
    T_w2c = image.cam_from_world

    # Preferred: pycolmap.Rigid3d has inverse().matrix()
    if hasattr(T_w2c, "inverse") and hasattr(T_w2c, "matrix"):
        return T_w2c.inverse().matrix()

    # Fallbacks (in case API differs)
    if hasattr(T_w2c, "matrix"):
        return np.linalg.inv(T_w2c.matrix())

    return np.linalg.inv(np.asarray(T_w2c))


def kabsch_umeyama(pts1: np.ndarray, pts2: np.ndarray, estimate_scale: bool = False):
    """
    Estimate similarity transform pts2 ≈ s * R * pts1 + t
    Returns:
        T (4x4): contains R, t (NO scale baked into R)
        s (float): scale
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    assert pts1.shape == pts2.shape and pts1.shape[1] == 3

    n = pts1.shape[0]
    mu1 = pts1.mean(axis=0)
    mu2 = pts2.mean(axis=0)
    x1 = pts1 - mu1
    x2 = pts2 - mu2

    # covariance for mapping pts1 -> pts2
    cov = (x1.T @ x2) / n  # 3x3

    U, D, Vt = np.linalg.svd(cov)

    # Ensure a proper rotation (det=+1)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1.0

    R = Vt.T @ S @ U.T  # pure rotation

    if estimate_scale:
        var1 = (np.sum(x1 ** 2) / n)
        # scale = trace(D * Sdiag) / var1
        Sdiag = np.diag(S)
        s = float(np.sum(D * Sdiag) / max(var1, 1e-12))
    else:
        s = 1.0

    t = mu2 - s * (R @ mu1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, s


def transform_poses_similarity(poses: np.ndarray, T: np.ndarray, scale: float) -> np.ndarray:
    """
    Apply similarity alignment to a batch of 4x4 poses (Nx4x4).
    If a pose is [R_i, t_i], output is:
        R'_i = R_align @ R_i
        t'_i = scale * (R_align @ t_i) + t_align
    """
    poses = np.asarray(poses, dtype=np.float64)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)

    R_align = T[:3, :3]
    t_align = T[:3, 3]

    out = poses.copy()

    # R' = R_align @ R_i
    out[:, :3, :3] = np.einsum("ij,njk->nik", R_align, poses[:, :3, :3])

    # t' = scale * (R_align @ t_i) + t_align
    Rt = np.einsum("ij,nj->ni", R_align, poses[:, :3, 3])
    out[:, :3, 3] = scale * Rt + t_align

    return out


def get_inliers(T_align, scale, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):
    # Apply similarity transform to GT poses to align into EST frame
    poses_gt_transformed = transform_poses_similarity(poses_gt, T_align, scale)

    translations_delta = poses_gt_transformed[:, :3, 3] - poses_est[:, :3, 3]
    rotations_delta = poses_gt_transformed[:, :3, :3] @ poses_est[:, :3, :3].transpose([0, 2, 1])

    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (inlier_threshold_r / 180.0 * math.pi)
    return np.logical_and(inliers_r, inliers_t)


def print_hyp(hypothesis, hyp_name):
    T = hypothesis["T"]
    s = hypothesis["scale"]
    h_translation = np.linalg.norm(T[:3, 3])
    h_angle = Rotation.from_matrix(T[:3, :3]).magnitude() * 180.0 / math.pi
    _logger.debug(
        f"{hyp_name}: score={hypothesis['score']}, "
        f"scale={s:.6f}, translation={h_translation:.3f}m, rotation={h_angle:.2f}deg."
    )


def estimate_alignment(
    estimates,
    min_cofident_estimates=10,   # keep original arg name for compatibility
    inlier_threshold_t=0.05,
    inlier_threshold_r=5,
    ransac_iterations=10000,
    refinement_max_hyp=12,
    refinement_max_it=8,
    estimate_scale=True
):
    _logger.debug("Estimate similarity transform between pose estimates and ground truth (T + scale separated).")

    # Filter invalid estimates (check both gt and est)
    valid_estimates = []
    for e in estimates:
        if e is None:
            continue
        if e.pose_gt is None or e.pose_est is None:
            continue
        if np.any(np.isnan(e.pose_gt)) or np.any(np.isinf(e.pose_gt)):
            continue
        if np.any(np.isnan(e.pose_est)) or np.any(np.isinf(e.pose_est)):
            continue
        valid_estimates.append(e)

    num_confident_estimates = len(valid_estimates)
    _logger.debug(f"{num_confident_estimates} estimates considered valid.")

    if num_confident_estimates < min_cofident_estimates:
        _logger.debug("Too few valid estimates. Aborting alignment.")
        return None, 1.0

    poses_est = np.empty((num_confident_estimates, 4, 4), dtype=np.float64)
    poses_gt = np.empty((num_confident_estimates, 4, 4), dtype=np.float64)
    for i, e in enumerate(valid_estimates):
        poses_est[i] = e.pose_est
        poses_gt[i] = e.pose_gt

    ransac_hypotheses = []
    min_sample_size = 3

    for _ in range(ransac_iterations):
        samples = random.sample(range(num_confident_estimates), min_sample_size)

        h_pts1 = poses_gt[samples, :3, 3]   # gt translations
        h_pts2 = poses_est[samples, :3, 3]  # est translations

        T_h, s_h = kabsch_umeyama(h_pts1, h_pts2, estimate_scale=estimate_scale)

        inliers = get_inliers(T_h, s_h, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

        # keep only hypotheses where minimal sample are all inliers
        if inliers[samples].sum() >= min_sample_size:
            ransac_hypotheses.append({
                "T": T_h,
                "scale": s_h,
                "inliers": inliers,
                "score": int(inliers.sum()),
            })

    if not ransac_hypotheses:
        _logger.debug("Did not find a single valid RANSAC hypothesis. Abort alignment estimation.")
        return None, 1.0

    ransac_hypotheses.sort(key=lambda x: x["score"], reverse=True)
    for i, hyp in enumerate(ransac_hypotheses[:min(len(ransac_hypotheses), 10)]):
        print_hyp(hyp, f"Hypothesis {i}")

    _logger.debug(f"Starting refinement of {refinement_max_hyp} best hypotheses.")
    ransac_hypotheses = ransac_hypotheses[:refinement_max_hyp]

    for ref_hyp in ransac_hypotheses:
        print_hyp(ref_hyp, "Pre-Refinement")

        for ref_it in range(refinement_max_it):
            inl = ref_hyp["inliers"]
            h_pts1 = poses_gt[inl, :3, 3]
            h_pts2 = poses_est[inl, :3, 3]

            T_new, s_new = kabsch_umeyama(h_pts1, h_pts2, estimate_scale=estimate_scale)
            inliers_new = get_inliers(T_new, s_new, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)
            refined_score = int(inliers_new.sum())

            if refined_score > ref_hyp["score"]:
                ref_hyp["T"] = T_new
                ref_hyp["scale"] = s_new
                ref_hyp["inliers"] = inliers_new
                ref_hyp["score"] = refined_score
                print_hyp(ref_hyp, f"Refinement iteration {ref_it}")
            else:
                _logger.debug(
                    f"Stopping refinement. Score did not improve: new={refined_score}, old={ref_hyp['score']}"
                )
                break

    ransac_hypotheses.sort(key=lambda x: x["score"], reverse=True)
    best = ransac_hypotheses[0]
    print_hyp(best, "Best")

    return best["T"], best["scale"]

def estimate_alignment_all(estimates, estimate_scale=True):
    valid = [e for e in estimates
             if e is not None and e.pose_gt is not None and e.pose_est is not None
             and not (np.any(np.isnan(e.pose_gt)) or np.any(np.isinf(e.pose_gt)))
             and not (np.any(np.isnan(e.pose_est)) or np.any(np.isinf(e.pose_est)))]
    if len(valid) < 3:
        return None, 1.0

    poses_est = np.stack([e.pose_est for e in valid], axis=0)
    poses_gt  = np.stack([e.pose_gt  for e in valid], axis=0)

    pts_gt  = poses_gt[:, :3, 3]
    pts_est = poses_est[:, :3, 3]

    T, s = kabsch_umeyama(pts_gt, pts_est, estimate_scale=estimate_scale)
    return T, s


from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

_logger = logging.getLogger(__name__)


def estimate_alignment_lm_robust(
    estimates,
    min_cofident_estimates: int = 3,
    estimate_scale: bool = True,
    sigma_t: float = 0.05,
    sigma_r_deg: float = 5.0,
    loss: str = "cauchy",                # "cauchy" / "huber" / "soft_l1" / "arctan" / "linear"
    f_scale: float = 1.0,
    max_nfev: int = 200,
    verbose: int = 0
):
    """Estimate a similarity transform aligning GT to estimates using robust nonlinear least squares. Parameters are [rvec(3), t(3), log_s(1)], omitting log_s when scale is fixed. R_align = Exp(rvec), s = exp(log_s). Translation residual: (s * (R_align @ t_gt) + t_align - t_est) / sigma_t. Rotation residual: Log((R_align @ R_gt) @ R_est.T) / sigma_r."""
    # ----------------------------
    # ----------------------------
    valid = []
    for e in estimates:
        if e is None or e.pose_gt is None or e.pose_est is None:
            continue
        if np.any(np.isnan(e.pose_gt)) or np.any(np.isinf(e.pose_gt)):
            continue
        if np.any(np.isnan(e.pose_est)) or np.any(np.isinf(e.pose_est)):
            continue
        valid.append(e)

    n = len(valid)
    _logger.debug(f"[LM] {n} estimates considered valid.")
    if n < min_cofident_estimates:
        _logger.debug("[LM] Too few valid estimates. Aborting alignment.")
        return None, 1.0

    poses_est = np.stack([e.pose_est for e in valid], axis=0).astype(np.float64)  # (N,4,4)
    poses_gt  = np.stack([e.pose_gt  for e in valid], axis=0).astype(np.float64)

    R_est = poses_est[:, :3, :3]
    t_est = poses_est[:, :3, 3]
    R_gt  = poses_gt[:, :3, :3]
    t_gt  = poses_gt[:, :3, 3]

    # ----------------------------
    # ----------------------------
    T0, s0 = kabsch_umeyama(t_gt, t_est, estimate_scale=estimate_scale)
    R0 = T0[:3, :3]
    t0 = T0[:3, 3]

    rvec0 = Rotation.from_matrix(R0).as_rotvec()
    if estimate_scale:
        s0_safe = float(s0) if float(s0) > 1e-12 else 1e-12
        x0 = np.concatenate([rvec0, t0, [math.log(s0_safe)]], axis=0)
    else:
        x0 = np.concatenate([rvec0, t0], axis=0)

    # ----------------------------
    # ----------------------------
    sigma_r = (sigma_r_deg / 180.0) * math.pi
    sigma_t = float(sigma_t)
    sigma_r = float(sigma_r)

    if sigma_t <= 0 or sigma_r <= 0:
        raise ValueError("sigma_t and sigma_r_deg must be > 0.")

    def residuals(x: np.ndarray) -> np.ndarray:
        rvec = x[0:3]
        t_align = x[3:6]
        if estimate_scale:
            s = float(np.exp(x[6]))
        else:
            s = 1.0

        R_align = Rotation.from_rotvec(rvec).as_matrix()  # (3,3)

        t_pred = (s * (t_gt @ R_align.T)) + t_align
        r_t = (t_pred - t_est) / sigma_t

        R_pred = np.einsum("ij,njk->nik", R_align, R_gt)                # (N,3,3)
        R_err  = np.einsum("nij,nkj->nik", R_pred, R_est)

        r_R = Rotation.from_matrix(R_err).as_rotvec() / sigma_r

        return np.concatenate([r_t.reshape(-1), r_R.reshape(-1)], axis=0)

    # ----------------------------
    # ----------------------------
    res = least_squares(
        residuals,
        x0,
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=verbose
    )

    x_opt = res.x
    rvec = x_opt[0:3]
    t_align = x_opt[3:6]
    if estimate_scale:
        s_opt = float(np.exp(x_opt[6]))
    else:
        s_opt = 1.0

    R_opt = Rotation.from_rotvec(rvec).as_matrix()
    T_opt = np.eye(4, dtype=np.float64)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_align

    _logger.debug(
        f"[LM] success={res.success}, cost={res.cost:.6f}, "
        f"scale={s_opt:.6f}, nfev={res.nfev}, status={res.status}"
    )
    return T_opt, s_opt

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def estimate_alignment_ransac_lm(
    estimates,
    min_cofident_estimates: int = 3,
    estimate_scale: bool = True,

    sigma_t: float = 0.05,
    sigma_r_deg: float = 5.0,

    loss: str = "cauchy",                # "cauchy" / "huber" / "soft_l1" / "arctan" / "linear"
    f_scale: float = 1.0,

    max_nfev: int = 200,
    verbose: int = 0,

    ransac_iters: int = 1000,
    inlier_thresh_t: float = 0.10,
    inlier_thresh_r_deg: float = 10.0,
    min_sample_size: int = 3,
    random_state: int = 0,

    refine_reestimate_inliers: bool = True,
):
    """Align GT to estimates with RANSAC followed by robust LM refinement. Model: t_est = s * (R_align @ t_gt) + t_align and R_est = R_align @ R_gt. Filter invalid poses, fit translation-based hypotheses, score translation/rotation inliers, and refine; optionally reclassify inliers and refine again. Return T_opt (4, 4, rotation/translation only), scale s_opt, a mask over valid samples, and an info dictionary."""

    # ----------------------------
    # ----------------------------
    valid = []
    valid_indices = []
    for i, e in enumerate(estimates):
        if e is None or e.pose_gt is None or e.pose_est is None:
            continue
        if np.any(np.isnan(e.pose_gt)) or np.any(np.isinf(e.pose_gt)):
            continue
        if np.any(np.isnan(e.pose_est)) or np.any(np.isinf(e.pose_est)):
            continue
        valid.append(e)
        valid_indices.append(i)

    n = len(valid)
    if n < max(min_cofident_estimates, min_sample_size):
        return None, 1.0, None, {
            "success": False,
            "reason": "too_few_valid_estimates",
            "num_valid": n,
        }

    poses_est = np.stack([e.pose_est for e in valid], axis=0).astype(np.float64)  # (N,4,4)
    poses_gt  = np.stack([e.pose_gt  for e in valid], axis=0).astype(np.float64)

    R_est = poses_est[:, :3, :3]
    t_est = poses_est[:, :3, 3]
    R_gt  = poses_gt[:, :3, :3]
    t_gt  = poses_gt[:, :3, 3]

    sigma_t = float(sigma_t)
    sigma_r = math.radians(float(sigma_r_deg))

    if sigma_t <= 0 or sigma_r <= 0:
        raise ValueError("sigma_t and sigma_r_deg must be > 0.")

    if inlier_thresh_t <= 0 or inlier_thresh_r_deg <= 0:
        raise ValueError("inlier_thresh_t and inlier_thresh_r_deg must be > 0.")

    sample_size = max(3, int(min_sample_size))
    if n < sample_size:
        return None, 1.0, None, {
            "success": False,
            "reason": "not_enough_samples_for_ransac",
            "num_valid": n,
            "sample_size": sample_size,
        }

    rng = np.random.default_rng(random_state)

    # ----------------------------
    # ----------------------------
    def _build_T(R_align: np.ndarray, t_align: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_align
        T[:3, 3] = t_align
        return T

    def _compute_errors(T_align: np.ndarray, s_align: float):
        """Return translation-error norms in the input units and rotation-error angles in degrees for all valid samples."""
        R_align = T_align[:3, :3]
        t_align = T_align[:3, 3]

        t_pred = s_align * (t_gt @ R_align.T) + t_align   # (N,3)
        trans_err = np.linalg.norm(t_pred - t_est, axis=1)

        R_pred = np.einsum("ij,njk->nik", R_align, R_gt)       # (N,3,3)
        R_err  = np.einsum("nij,nkj->nik", R_pred, R_est)      # R_pred @ R_est^T
        rotvec = Rotation.from_matrix(R_err).as_rotvec()       # (N,3)
        rot_err_deg = np.degrees(np.linalg.norm(rotvec, axis=1))

        return trans_err, rot_err_deg

    def _is_degenerate(indices: np.ndarray) -> bool:
        """Reject degenerate minimal samples with nearly coincident or low-rank GT or estimated positions."""
        pts_gt = t_gt[indices]
        pts_est_ = t_est[indices]

        pts_gt_c = pts_gt - pts_gt.mean(axis=0, keepdims=True)
        pts_est_c = pts_est_ - pts_est_.mean(axis=0, keepdims=True)

        rank_gt = np.linalg.matrix_rank(pts_gt_c)
        rank_est = np.linalg.matrix_rank(pts_est_c)

        return (rank_gt < 2) or (rank_est < 2)

    def _lm_refine(mask: np.ndarray, T_init: np.ndarray, s_init: float):
        """Refine only the masked inliers with LM."""
        if mask.sum() < min_cofident_estimates:
            return None, None, None

        R_gt_in = R_gt[mask]
        t_gt_in = t_gt[mask]
        R_est_in = R_est[mask]
        t_est_in = t_est[mask]

        R0 = T_init[:3, :3]
        t0 = T_init[:3, 3]
        rvec0 = Rotation.from_matrix(R0).as_rotvec()

        if estimate_scale:
            s0_safe = float(s_init) if float(s_init) > 1e-12 else 1e-12
            x0 = np.concatenate([rvec0, t0, [math.log(s0_safe)]], axis=0)
        else:
            x0 = np.concatenate([rvec0, t0], axis=0)

        def residuals(x: np.ndarray) -> np.ndarray:
            rvec = x[0:3]
            t_align = x[3:6]
            if estimate_scale:
                s = float(np.exp(x[6]))
            else:
                s = 1.0

            R_align = Rotation.from_rotvec(rvec).as_matrix()

            t_pred = s * (t_gt_in @ R_align.T) + t_align
            r_t = (t_pred - t_est_in) / sigma_t

            R_pred = np.einsum("ij,njk->nik", R_align, R_gt_in)
            R_err  = np.einsum("nij,nkj->nik", R_pred, R_est_in)
            r_R = Rotation.from_matrix(R_err).as_rotvec() / sigma_r

            return np.concatenate([r_t.reshape(-1), r_R.reshape(-1)], axis=0)

        res = least_squares(
            residuals,
            x0,
            method="trf",
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            verbose=verbose,
        )

        x_opt = res.x
        rvec = x_opt[0:3]
        t_align = x_opt[3:6]
        if estimate_scale:
            s_opt = float(np.exp(x_opt[6]))
        else:
            s_opt = 1.0

        R_opt = Rotation.from_rotvec(rvec).as_matrix()
        T_opt = _build_T(R_opt, t_align)
        return T_opt, s_opt, res

    # ----------------------------
    # ----------------------------
    best = None
    best_inlier_count = -1
    best_inlier_cost = np.inf

    for _ in range(ransac_iters):
        indices = rng.choice(n, size=sample_size, replace=False)

        if _is_degenerate(indices):
            continue

        try:
            T_h, s_h = kabsch_umeyama(
                t_gt[indices],
                t_est[indices],
                estimate_scale=estimate_scale
            )
        except Exception:
            continue

        trans_err, rot_err_deg = _compute_errors(T_h, s_h)

        inlier_mask = (trans_err <= inlier_thresh_t) & (rot_err_deg <= inlier_thresh_r_deg)
        inlier_count = int(inlier_mask.sum())

        if inlier_count < min_cofident_estimates:
            continue

        norm_cost = np.sum(
            (trans_err[inlier_mask] / inlier_thresh_t) ** 2 +
            (rot_err_deg[inlier_mask] / inlier_thresh_r_deg) ** 2
        )

        better = False
        if inlier_count > best_inlier_count:
            better = True
        elif inlier_count == best_inlier_count and norm_cost < best_inlier_cost:
            better = True

        if better:
            best = {
                "T": T_h,
                "s": s_h,
                "inlier_mask": inlier_mask,
                "trans_err": trans_err,
                "rot_err_deg": rot_err_deg,
            }
            best_inlier_count = inlier_count
            best_inlier_cost = norm_cost

    if best is None:
        return None, 1.0, None, {
            "success": False,
            "reason": "ransac_failed",
            "num_valid": n,
            "ransac_iters": ransac_iters,
        }

    # ----------------------------
    # ----------------------------
    inlier_mask = best["inlier_mask"]

    T0, s0 = kabsch_umeyama(
        t_gt[inlier_mask],
        t_est[inlier_mask],
        estimate_scale=estimate_scale
    )

    T_opt, s_opt, res = _lm_refine(inlier_mask, T0, s0)
    if T_opt is None:
        return None, 1.0, None, {
            "success": False,
            "reason": "lm_refine_failed_on_ransac_inliers",
            "num_valid": n,
            "num_inliers": int(inlier_mask.sum()),
        }

    # ----------------------------
    # ----------------------------
    trans_err, rot_err_deg = _compute_errors(T_opt, s_opt)
    final_inlier_mask = (trans_err <= inlier_thresh_t) & (rot_err_deg <= inlier_thresh_r_deg)

    if refine_reestimate_inliers and final_inlier_mask.sum() >= min_cofident_estimates:
        if not np.array_equal(final_inlier_mask, inlier_mask):
            T1, s1 = kabsch_umeyama(
                t_gt[final_inlier_mask],
                t_est[final_inlier_mask],
                estimate_scale=estimate_scale
            )
            T_refined, s_refined, res2 = _lm_refine(final_inlier_mask, T1, s1)
            if T_refined is not None:
                T_opt, s_opt = T_refined, s_refined
                res = res2

                trans_err, rot_err_deg = _compute_errors(T_opt, s_opt)
                final_inlier_mask = (trans_err <= inlier_thresh_t) & (rot_err_deg <= inlier_thresh_r_deg)

    info = {
        "success": True,
        "num_valid": n,
        "num_inliers": int(final_inlier_mask.sum()),
        "inlier_ratio": float(final_inlier_mask.mean()),
        "ransac_iters": ransac_iters,
        "ransac_best_inliers": int(best_inlier_count),
        "lm_success": bool(getattr(res, "success", True)),
        "lm_cost": float(getattr(res, "cost", 0.0)),
        "lm_nfev": int(getattr(res, "nfev", 0)),
        "scale": float(s_opt),
        "median_trans_err_inliers": float(np.median(trans_err[final_inlier_mask])) if final_inlier_mask.any() else float("inf"),
        "median_rot_err_deg_inliers": float(np.median(rot_err_deg[final_inlier_mask])) if final_inlier_mask.any() else float("inf"),
        "valid_indices": valid_indices,
    }

    return T_opt, s_opt, final_inlier_mask, info
