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
import torch.nn.functional as F

from poses.matcher import Matches
from poses.loma_backend import get_loma_match_utils

class DescribedKeypoints():
    """
    A class to store 2D keypoints, their descriptors, their matches to other images, and their estimated 3D positions.
    """
    @torch.no_grad()
    def __init__(self, kpts, feats, norm_kpts=None):
        self.kpts = kpts
        self.feats = feats
        self.norm_kpts = norm_kpts
        self.valid = feats.abs().sum(dim=-1) > 0
        self.nvalid = self.valid.sum()
        self.has_pt3d = torch.zeros(kpts.shape[0], dtype=torch.bool, device=kpts.device)
        self.pts_conf = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device)
        self.pts3d = torch.zeros(kpts.shape[0], 3, dtype=torch.float, device=kpts.device)
        self.depth = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device)
        self.matches = {}

    @torch.no_grad()
    def update_matches(self, other_id, matches, swap=False):
        if swap:
            matches = Matches(
                matches.kpts_other,
                matches.kpts,
                matches.idx_other,
                matches.idx,
                matches.scores,
            )
        self.matches[other_id] = matches
 
    @torch.no_grad()
    def update_3D_pts(self, pts3D, depth, conf, idx):
        self.has_pt3d[idx] = True
        self.depth[idx] = depth
        self.pts3d[idx] = pts3D
        self.pts_conf[idx] = conf

    @torch.no_grad()
    def clear_3D_pts(self):
        """Invalidate every cached 3D estimate before pose-based retriangulation."""
        self.has_pt3d.zero_()
        self.depth.zero_()
        self.pts3d.zero_()
        self.pts_conf.zero_()

    @torch.no_grad()
    def to(self, device):
        self.kpts = self.kpts.to(device)
        self.feats = self.feats.to(device)
        if self.norm_kpts is not None:
            self.norm_kpts = self.norm_kpts.to(device)
        self.valid = self.valid.to(device)
        self.has_pt3d = self.has_pt3d.to(device)
        self.pts_conf = self.pts_conf.to(device)
        self.pts3d = self.pts3d.to(device)
        self.depth = self.depth.to(device)
        for key in self.matches:
            self.matches[key].kpts = self.matches[key].kpts.to(device)
            self.matches[key].kpts_other = self.matches[key].kpts_other.to(device)
            self.matches[key].idx = self.matches[key].idx.to(device)
            self.matches[key].idx_other = self.matches[key].idx_other.to(device)
            if self.matches[key].scores is not None:
                self.matches[key].scores = self.matches[key].scores.to(device)

class Detector():
    @torch.no_grad()
    def __init__(self, top_k, width, height, backend="lomag", loma_model=None):
        self.backend = backend
        self.loma_model = loma_model
        if self.loma_model is None:
            raise ValueError("LoMa detector backend requires a LoMa model.")
        _, self.loma_to_pixel_coords = get_loma_match_utils()
        return

    @torch.no_grad()
    def __call__(self, image):
        orig_height, orig_width = image.shape[-2:]
        loma_height = ((orig_height + 13) // 14) * 14
        loma_width = ((orig_width + 13) // 14) * 14
        loma_image = image
        if loma_height != orig_height or loma_width != orig_width:
            loma_image = F.interpolate(
                image[None],
                size=(loma_height, loma_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        norm_kpts, feats, height, width = self.loma_model.detect_and_describe(
            loma_image[None], self.loma_model.cfg.num_keypoints
        )
        kpts = self.loma_to_pixel_coords(norm_kpts[0], orig_height, orig_width)
        return DescribedKeypoints(kpts.float(), feats[0].float(), norm_kpts[0].float())

