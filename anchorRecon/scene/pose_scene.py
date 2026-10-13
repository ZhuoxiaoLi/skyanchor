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
from poses.feature_detector import DescribedKeypoints
from scene.keyframe import Keyframe


class PoseScene:
    """Keyframe collection and spatial reference selection for anchored poses."""

    def __init__(self, width, height, args, matcher):
        self.width = width
        self.height = height
        self.matcher = matcher
        self.num_prev_keyframes_check = args.num_prev_keyframes_check
        self.keyframe_selection_use_3d = args.keyframe_selection_use_3d
        self.keyframes = []
        self.f = 0.7 * width
        self.approx_cam_centres = None
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
        self.sorted_frame_indices = None

    @torch.no_grad()
    def get_prev_keyframes(self, n: int, update_3dpts: bool, desc_kpts: DescribedKeypoints = None):
        """
        Get the n previous keyframes that are the closest to the last
        If desc_kpts is not None, we find the previous keyframes that have the most matches with desc_kpts. The search window is given by self.num_prev_keyframes_check
        """

        # Keep this helper safe for programmatic callers as well as CLI users.
        # Ranking cannot select more entries than either the registered set or
        # the candidate window.
        n = min(max(int(n), 0), len(self.keyframes))
        if n == 0:
            return []

        # Look for the previous keyframes with the most matches with desc_kpts (if provided)
        if desc_kpts is not None and len(self.keyframes) > n:
            n_ckecks = min(
                max(self.num_prev_keyframes_check, n), len(self.keyframes)
            )
            keyframes_indices_to_check = self.sorted_frame_indices[:n_ckecks]
            n_matches = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(
                    self.keyframes[index].desc_kpts,
                    desc_kpts,
                    require_3d=bool(getattr(self, "keyframe_selection_use_3d", False)),
                )
            _, top_indices = torch.topk(n_matches, n)
            prev_keyframes_indices = keyframes_indices_to_check[top_indices.cpu()]
        # If desc_kpts is not provided, we take the n closest keyframes
        else:
            prev_keyframes_indices = self.sorted_frame_indices[:n]
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # Re-run triangulation if necessary
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes


    def refresh_pose_cache(self):
        """Rebuild cached poses and approximate camera centers after an external pose update."""
        if len(self.keyframes) == 0:
            self.approx_cam_centres = None
            self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
            self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
            self.sorted_frame_indices = None
            return

        Rts = torch.stack([keyframe.get_Rt() for keyframe in self.keyframes], dim=0)
        self.cached_Rts = Rts
        self.valid_Rt_cache = torch.ones(len(self.keyframes), device="cuda", dtype=torch.bool)
        self.approx_cam_centres = torch.stack(
            [keyframe.get_centre(approx=True) for keyframe in self.keyframes], dim=0
        )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - self.approx_cam_centres[-1:],
            dim=-1,
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()


    def add_keyframe(self, keyframe: Keyframe, f=None):
        """Register a pose and update spatial ordering and pose caches."""


        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update intrinsics
        if f is not None:
            self.f = f.item()

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, torch.ones(1, device="cuda", dtype=torch.bool)], dim=0
        )

