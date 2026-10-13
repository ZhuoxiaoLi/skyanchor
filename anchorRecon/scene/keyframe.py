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

from __future__ import annotations
from argparse import Namespace
import torch

from poses.feature_detector import DescribedKeypoints
from poses.triangulator import Triangulator
from utils import (
    sixD2mtx,
    principal_point_tensor,
)
from dataloaders.read_write_model import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    Image, camera pose and matched observations used by triangulation and BA.
    """
    def __init__(
        self,
        image: torch.Tensor,
        info: dict,
        desc_kpts: DescribedKeypoints,
        Rt: torch.Tensor,
        index: int,
        f: torch.Tensor,
        triangulator: Triangulator,
        args: Namespace,
        lock_pose: bool = False,
    ):
        principal_point = info.get("principal_point", None)
        if principal_point is not None and not torch.is_tensor(principal_point):
            principal_point = torch.tensor(principal_point, dtype=torch.float32)

        self.image_pyr = [image]
        # Shared geometry for triangulation, pose estimation and final submap BA.
        self.width = image.shape[2]
        self.height = image.shape[1]
        if principal_point is not None:
            self.centre = principal_point.to(device="cuda", dtype=torch.float32)
        else:
            self.centre = principal_point_tensor(
                self.width,
                self.height,
                args.init_cx if args is not None else None,
                args.init_cy if args is not None else None,
                device="cuda",
            )
        self.f = f
        self.triangulator = triangulator
        self.index = index


        self.desc_kpts = desc_kpts
        self.info = info
        self.is_test = info["is_test"]
        self.lock_pose = lock_pose

        # Optimizable parameters
        self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone())
        self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone())
        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def to(self, device: str):
        self.image_pyr = [image.to(device) for image in self.image_pyr]

    @property
    def device(self):
        return self.image_pyr[0].device

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()

    @torch.no_grad()
    def update_3dpts(self, all_keyframes: list[Keyframe]):
        """
        Update keypoint 3D positions by triangulating matched observations. 
        """
        unload_desc_kpts = self.desc_kpts.kpts.device.type == "cpu"
        if unload_desc_kpts:
            self.desc_kpts.to("cuda")

        ## Triangulation
        # Select keyframes to triangulate with based on their locations
        uv, uvs_others, chosen_kfs_ids = self.triangulator.prepare_matches(
            self.desc_kpts
        )
        valid_chosen = [
            (row, index)
            for row, index in enumerate(chosen_kfs_ids)
            if 0 <= index < len(all_keyframes)
        ]
        if len(valid_chosen) == 0:
            if unload_desc_kpts:
                self.desc_kpts.to("cpu")
            return

        filtered_uvs_others = -torch.ones_like(uvs_others)
        filtered_chosen_kfs_ids = []
        for out_row, (in_row, index) in enumerate(valid_chosen[: self.triangulator.n_cams]):
            filtered_uvs_others[out_row] = uvs_others[in_row]
            filtered_chosen_kfs_ids.append(index)
        uvs_others = filtered_uvs_others
        Rts_others = torch.stack(
            [all_keyframes[index].get_Rt() for index in filtered_chosen_kfs_ids],
            dim=0,
        )
        if len(Rts_others) < self.triangulator.n_cams:
            Rts_others = torch.cat(
                [
                    Rts_others,
                    torch.eye(4, device="cuda")[None].repeat(
                        self.triangulator.n_cams - len(Rts_others), 1, 1
                    ),
                ],
                dim=0,
            )

        # Run the triangulator and update the 3D points
        new_pts, depth, best_dis, valid_matches = self.triangulator(
            uv, uvs_others, self.get_Rt(), Rts_others, self.f, self.centre
        )
        self.desc_kpts.update_3D_pts(
            new_pts[valid_matches], depth[valid_matches], 1, valid_matches
        )

        if unload_desc_kpts:
            self.desc_kpts.to("cpu")









    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        # first param of params is focal length in pixels
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[self.f.item(), self.centre[0].item(), self.centre[1].item()],
        )

        image = BaseImage(
            id=id,
            name=self.info.get("name", str(id)),
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
