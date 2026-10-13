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
from enum import Enum
from poses.mini_ba import MiniBA
from utils import pts2px, sixD2mtx


class EstimatorType(Enum):
    P4P = 1


class RANSACEstimator:
    @torch.no_grad()
    def __init__(self, N: int, max_error: float, type: EstimatorType):
        """
        Initialize the RANSAC estimator.

        Args:
            N (int): Number of models to estimate.
            max_error (float): Maximum reprojection error for inliers.
            type (EstimatorType): Type of estimator to use.
        """
        self.N = N
        self.max_error = max_error
        self.type = type

        # Set the functions and number of points required for each estimator
        if type == EstimatorType.P4P:
            self.m = 4  # 4 pairs are required per estimation
            self.model_estimator = MiniBA(
                self.N,
                1,
                0,
                self.m,
                optimize_focal=False,
                optimize_3Dpts=False,
                make_cuda_graph=True,
                outlier_mad_scale=0,
                iters=20,
            )
            self.models = torch.zeros([N, 3, 4], device=torch.device("cuda"))
        else:
            raise ValueError(f"Unknown EstimatorType {type}")

    def estimate(
        self,
        mkpts1: torch.Tensor,
        mkpts2: torch.Tensor,
        idxs: torch.Tensor,
        focal: torch.Tensor,
        centre: torch.Tensor,
        R6D_init: torch.Tensor,
        t_init: torch.Tensor,
    ):
        """
        Estimate N models from the given matches.
        """
        if self.type == EstimatorType.P4P:
            mkpts1_candidates = mkpts1[idxs].view(self.N, -1)
            mkpts2_candidates = mkpts2[idxs]
            Rs6D_init = R6D_init[None, None].repeat(self.N, 1, 1, 1)
            ts_init = t_init[None, None].repeat(self.N, 1, 1)
            Rs6D, ts, _, _, _, _, _ = self.model_estimator(
                Rs6D_init, ts_init, focal, mkpts2_candidates, centre, mkpts1_candidates
            )
            self.models[..., :3] = sixD2mtx(Rs6D).squeeze()
            self.models[..., 3] = ts.squeeze()

    def get_inlier_mask(self, mkpts1, mkpts2, focal, centre):
        if self.type == EstimatorType.P4P:
            mkpts2_cam = (
                torch.matmul(mkpts2, self.models[..., :3].transpose(-2, -1))
                + self.models[..., None, :, 3]
            )
            mkpts2_px = pts2px(mkpts2_cam, focal, centre)
            inliers = (
                torch.linalg.vector_norm(mkpts1[None] - mkpts2_px, dim=-1)
                < self.max_error
            ) & torch.isfinite(mkpts2_cam).all(dim=-1) & (mkpts2_cam[..., 2] > 1e-6)
        return inliers

    @torch.no_grad()
    def __call__(
        self,
        mkpts1,
        mkpts2,
        focal=None,
        centre=None,
        R6D_init=None,
        t_init=None,
        confs=None,
    ):
        """
        Run the RANSAC estimator to find the best model and inliers.
        args:
            mkpts1: (n, 2) 2D positions of the matched keypoints
            mkpts2: (n, 3) 3D points for PnP
            focal: (1) for PnP, else None
            centre: (2) for PnP, else None
            R6D_init: (3, 2) initial 6D rotation for PnP, else None
            t_init: (3) initial translation for PnP, else None
            confs: (3) confidence to scale the weights of the inliers

        returns:
            best_model: (3, 4) world-to-camera pose
            mask: (n) boolean mask of the inliers
        """

        if self.type == EstimatorType.P4P:
            assert focal is not None
            assert centre is not None

        # Select N x m random points
        random_scores = torch.rand(self.N, mkpts1.shape[0], device=mkpts1.device)
        _, idxs = torch.topk(random_scores, self.m, dim=1)

        # Run the batch estimator
        self.estimate(mkpts1, mkpts2, idxs, focal, centre, R6D_init, t_init)

        # Get the inlier mask
        inliers = self.get_inlier_mask(mkpts1, mkpts2, focal, centre)

        # Compute inlier mask and find the best model
        if confs is not None:
            inliers = inliers * confs[None]
        n_inliers = inliers.sum(dim=1)
        best_id = torch.argmax(n_inliers)
        best_model = self.models[best_id]
        mask = inliers[best_id] > 0

        return best_model, mask
