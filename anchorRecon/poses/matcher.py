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

import torch
import cv2

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poses.feature_detector import DescribedKeypoints
from poses.loma_backend import get_loma_match_utils
from utils import get_global_profiler


class Matches:
    """
    A class to store matched keypoints and their indices between two sets of keypoints.
    """
    def __init__(self, kpts, kpts_other, idx, idx_other, scores=None):
        self.kpts = kpts
        self.kpts_other = kpts_other
        self.idx = idx
        self.idx_other = idx_other
        self.scores = scores


# Adapted from https://github.com/verlab/accelerated_features
def match(feats1, feats2, min_cossim=0.82):
    cossim = feats1 @ feats2.t()

    bestcossim, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)

    idx0 = torch.arange(match12.shape[0], device=match12.device)
    mask = match21[match12] == idx0

    if min_cossim > 0:
        mask *= bestcossim > min_cossim

    return idx0, match12, mask, bestcossim


class Matcher:
    @torch.no_grad()
    def __init__(
        self,
        min_cossim: float = 0.82,
        backend: str = "lomag",
        loma_model=None,
        loma_filter_threshold: float = -1.0,
        loma_magsac_threshold: float = 0.2,
        loma_magsac_confidence: float = 0.999999,
        loma_magsac_max_iters: int = 10000,
    ):
        """LoMa matching with MAGSAC filtering and cheap reference ranking."""
        self.min_cossim = min_cossim
        self.backend = backend
        self.loma_model = loma_model
        self.loma_filter_threshold = loma_filter_threshold
        self.loma_magsac_threshold = loma_magsac_threshold
        self.loma_magsac_confidence = loma_magsac_confidence
        self.loma_magsac_max_iters = loma_magsac_max_iters
        self.last_match_stats = {}
        if self.loma_model is None:
            raise ValueError("LoMa matcher backend requires a LoMa model.")
        self.loma_filter_matches, _ = get_loma_match_utils()

    def evaluate_match(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        require_3d: bool = False,
    ):
        """
        Get the number of matches between two sets of described keypoints.
        """
        with get_global_profiler().stage("match.evaluate_cossim"):
            # Keep keyframe selection cheap. Exact LoMa matching is used in __call__.
            idx, _, mask, _ = match(
                desc_kpts.feats.cuda(),
                desc_kpts_other.feats.cuda(),
                self.min_cossim,
            )
            if require_3d:
                has_pt3d = getattr(desc_kpts, "has_pt3d", None)
                if has_pt3d is None:
                    return torch.zeros((), device=mask.device, dtype=torch.long)
                return (mask & has_pt3d.to(mask.device)[idx]).sum()
            return mask.sum()

    @torch.no_grad()
    def __call__(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        remove_outliers: bool = False,
        update_kpts_flag: str = "",
        kID: int = -1,
        kID_other: int = -1,
    ):
        """
        Matches keypoints between two sets of described keypoints, with optional outlier removal based on the fundamental RANSAC estimation.
        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            remove_outliers (bool): Whether to remove outliers using the fundamental matrix.
            update_kpts_flag (str): If "all", updates all matches; if "inliers", updates only inliers.
            kID (int): ID of the first set of keypoints, used for updating matches.
            kID_other (int): ID of the second set of keypoints, used for updating matches.
        Returns:
            Matches: A Matches object containing the matched keypoints and their indices.
        """
        prof = get_global_profiler()
        if desc_kpts.norm_kpts is None or desc_kpts_other.norm_kpts is None:
            raise ValueError("LoMa matching requires normalized LoMa keypoints.")
        with prof.stage("match.loma"):
            scores = self.loma_model(
                desc_kpts.norm_kpts[None],
                desc_kpts_other.norm_kpts[None],
                desc_kpts.feats[None],
                desc_kpts_other.feats[None],
            )["scores"]
            filter_threshold = (
                self.loma_model.cfg.filter_threshold
                if self.loma_filter_threshold < 0
                else self.loma_filter_threshold
            )
            m0, _, mscores0, _ = self.loma_filter_matches(scores, filter_threshold)
            valid = m0[0] > -1
            idx = torch.where(valid)[0]
            idx_other = m0[0][valid]
            match_scores = mscores0[0][valid]

        raw_match_count = len(idx)
        kpts = desc_kpts.kpts[idx]
        kpts_other = desc_kpts_other.kpts[idx_other]
        idx_all = idx
        idx_other_all = idx_other
        kpts_all = kpts
        kpts_other_all = kpts_other
        match_scores_all = match_scores

        if remove_outliers:
            with prof.stage("match.magsac"):
                mask = self._magsac_inlier_mask(kpts, kpts_other)
            idx = idx[mask]
            idx_other = idx_other[mask]
            kpts = kpts[mask]
            kpts_other = kpts_other[mask]
            match_scores = match_scores[mask]

        self.last_match_stats = {
            "backend": self.backend,
            "kID": kID,
            "kID_other": kID_other,
            "raw_matches": int(raw_match_count),
            "geom_inliers": int(len(idx)),
            "remove_outliers": bool(remove_outliers),
            "magsac_threshold": float(self.loma_magsac_threshold),
            "mean_score": float(match_scores.mean().item()) if len(match_scores) > 0 else 0.0,
        }

        if update_kpts_flag == "all":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts_all, kpts_other_all, idx_all, idx_other_all, match_scores_all)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other_all, kpts_all, idx_other_all, idx_all, match_scores_all)
            )
        elif update_kpts_flag == "inliers":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts, kpts_other, idx, idx_other, match_scores)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other, kpts, idx_other, idx, match_scores)
            )

        return Matches(kpts, kpts_other, idx, idx_other, match_scores)

    def _magsac_inlier_mask(self, kpts, kpts_other):
        if len(kpts) < 8:
            return torch.zeros(len(kpts), dtype=torch.bool, device=kpts.device)

        kpts_np = kpts.detach().float().cpu().numpy()
        kpts_other_np = kpts_other.detach().float().cpu().numpy()
        method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
        _, mask_np = cv2.findFundamentalMat(
            kpts_np,
            kpts_other_np,
            ransacReprojThreshold=self.loma_magsac_threshold,
            method=method,
            confidence=self.loma_magsac_confidence,
            maxIters=self.loma_magsac_max_iters,
        )
        if mask_np is None:
            return torch.zeros(len(kpts), dtype=torch.bool, device=kpts.device)
        return torch.from_numpy(mask_np.reshape(-1).astype(bool)).to(kpts.device)
