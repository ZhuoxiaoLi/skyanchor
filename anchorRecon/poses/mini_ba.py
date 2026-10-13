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

from utils import mtx2sixD, pts2px, sixD2mtx

import torch
import torch.nn as nn
from torch.func import vmap, jacfwd


def project(xyz, R6D_t, f, centre):
    R6D = R6D_t[:6].reshape(3, 2)
    t = R6D_t[6:9]
    R = sixD2mtx(R6D)
    xyz_local = R @ xyz + t
    return pts2px(xyz_local, f, centre)


def get_residual(xyz, R6D_t, f, centre, uv):
    projected = project(xyz, R6D_t, f, centre)
    return projected - uv


def get_residual2(xyz, R6D_t, f, centre, uv):
    err = get_residual(xyz, R6D_t, f, centre, uv)
    return err, err


class MiniBAInternal(nn.Module):
    def __init__(
        self,
        batch,
        n_opt_cams,
        n_fixed_cams,
        npts,
        optimize_focal,
        optimize_3Dpts,
        huber_delta,
        outlier_mad_scale,
        lm,
        ep,
        k,
        iters,
    ):
        super().__init__()
        self.batch = batch
        self.n_opt_cams = n_opt_cams
        self.n_fixed_cams = n_fixed_cams
        self.n_cams = n_opt_cams + n_fixed_cams
        self.npts = npts
        self.optimize_focal = optimize_focal
        self.optimize_3Dpts = optimize_3Dpts
        self.huber_delta = huber_delta
        self.outlier_mad_scale = outlier_mad_scale
        self.lm = lm
        self.ep = ep
        self.k = k
        self.iters = iters
        self.n_cam_params = n_opt_cams * 9 + optimize_focal

        # Optimized parameters and their index in get_residual2 args (for autodiff)
        argnums = (1,)
        self.param2id_dict = {"poses": len(argnums) - 1}
        if optimize_focal:
            argnums += (2,)
            self.param2id_dict["focal"] = len(argnums) - 1
        if optimize_3Dpts:
            argnums += (0,)
            self.param2id_dict["xyz"] = len(argnums) - 1

        # Prepare projection and jacobian estimation function with autodiff
        self.get_residual_jacobian = vmap(
            vmap(jacfwd(get_residual2, has_aux=True, argnums=argnums))
        )
        self.get_residual = vmap(vmap(get_residual))

    def prepare_for_proj(self, xyz, Rs6D, ts, f, centre):
        """
        Expand and organize for jacobian computation
        """
        xyz_e = xyz.unsqueeze(1).expand(-1, self.n_cams, *xyz.shape[1:])
        Rs6D_ts = torch.cat([Rs6D.view(-1, 6), ts], dim=-1)
        Rs6D_ts_e = Rs6D_ts[None].expand(self.npts, *Rs6D_ts.shape)
        f_e = f[None, None].expand(self.npts, self.n_cams, *f.shape)
        centre_e = centre[None, None].expand(self.npts, self.n_cams, *centre.shape)

        return xyz_e, Rs6D_ts_e, f_e, centre_e

    def get_mask(self, r_in, original_mask2):
        if self.outlier_mad_scale > 0:
            # Per-camera robust threshold over valid observations only.  The
            # old quantile call omitted ``dim`` and therefore flattened every
            # camera/point into one distribution while interpreting the
            # camera-dependent q vector as several global quantiles.
            r_grid = r_in.view(self.npts, self.n_cams, 2)
            safe_r = torch.where(
                original_mask2[..., None], r_grid, torch.zeros_like(r_grid)
            )
            err = torch.linalg.vector_norm(safe_r, dim=-1)
            nan = torch.full_like(err, float("nan"))
            masked_err = torch.where(original_mask2, err, nan)
            med = torch.nanmedian(masked_err, dim=0).values
            abs_dev = torch.abs(err - med[None])
            mad = torch.nanmedian(
                torch.where(original_mask2, abs_dev, nan), dim=0
            ).values
            c = med + self.outlier_mad_scale * mad
            c = torch.nan_to_num(c, nan=5.0, posinf=5.0, neginf=5.0)
            c.clamp_min_(5.0)

            # get mask
            mask = original_mask2 & (err < c[None])
            mask = mask[..., None].expand(-1, -1, 2).reshape(-1).float()
            return mask
        else:
            return original_mask2[..., None].expand(-1, -1, 2).reshape(-1).float()

    def get_huber_weights(self, r):
        if self.huber_delta > 0:
            r_abs = r.abs()
            weights = torch.where(
                r_abs <= self.huber_delta, 1, self.huber_delta / r_abs.sqrt()
            )
            return weights
        else:
            return torch.ones_like(r)

    def get_positive_depth_mask(self, xyz, Rs6D, ts):
        """Return per-point/per-camera cheirality for the current state."""
        R = sixD2mtx(Rs6D)
        xyz_cam = torch.einsum("cij,pj->pci", R, xyz) + ts[None]
        return torch.isfinite(xyz_cam).all(dim=-1) & (xyz_cam[..., 2] > 1e-6)

    def optimize(self, Rs6D, ts, f, xyz, centre, uv):
        uv = uv.view(self.npts, self.n_cams, 2)
        observation_mask = (uv >= 0).all(dim=-1)
        lm = self.lm

        for iteration in range(self.iters):
            # Compute Jacobian and residuals
            jacobian_elements, r_in = self.get_residual_jacobian(
                *self.prepare_for_proj(xyz, Rs6D, ts, f, centre), uv
            )
            r_in = r_in.view(-1)
            # Jacobian w.r.t the camera poses
            duv_dcam = jacobian_elements[self.param2id_dict["poses"]]
            duv_dcam = duv_dcam.unsqueeze(-2).repeat(1, 1, 1, self.n_opt_cams, 1)
            j = torch.arange(duv_dcam.shape[1], device=f.device).view(1, -1, 1, 1, 1)
            l = torch.arange(duv_dcam.shape[3], device=f.device).view(1, 1, 1, -1, 1)
            mask = (j != l).expand_as(duv_dcam)
            duv_dcam = torch.where(mask, 0, duv_dcam)
            duv_dcam = duv_dcam.reshape(*duv_dcam.shape[:-2], -1)
            # Jacobian w.r.t the focal length
            if self.optimize_focal:
                duv_dcam = torch.cat(
                    [duv_dcam, jacobian_elements[self.param2id_dict["focal"]]], dim=-1
                )
            # Jacobian w.r.t the 3D points
            if self.optimize_3Dpts:
                duv_dxyz = jacobian_elements[self.param2id_dict["xyz"]]

            if iteration == 0:
                initial_r = r_in.clone()

            # Robustification
            positive_depth = self.get_positive_depth_mask(xyz, Rs6D, ts)
            finite_residual = torch.isfinite(
                r_in.view(self.npts, self.n_cams, 2)
            ).all(dim=-1)
            valid_observations = observation_mask & positive_depth & finite_residual
            mask = self.get_mask(r_in, valid_observations)
            active_coordinates = mask > 0
            safe_r_in = torch.where(
                active_coordinates, r_in, torch.zeros_like(r_in)
            )
            weights = self.get_huber_weights(safe_r_in)
            r = torch.where(
                active_coordinates,
                safe_r_in * weights,
                torch.zeros_like(safe_r_in),
            )
            jacobian_weights = weights.reshape(
                self.npts, self.n_cams, 2, 1
            )
            active_jacobian = active_coordinates.reshape(
                self.npts, self.n_cams, 2, 1
            )
            duv_dcam = torch.where(
                active_jacobian,
                duv_dcam * jacobian_weights,
                torch.zeros_like(duv_dcam),
            )
            if self.optimize_3Dpts:
                duv_dxyz = torch.where(
                    active_jacobian,
                    duv_dxyz * jacobian_weights,
                    torch.zeros_like(duv_dxyz),
                )

            duv_dcam_flat = duv_dcam.reshape(-1, self.n_cam_params)
            duv_dcam_reshaped = duv_dcam.reshape(
                self.npts, self.n_cams * 2, self.n_cam_params
            )

            # Precompute each block of Jacobian^T @ Jacobian
            jtj_cam = duv_dcam_flat.T @ duv_dcam_flat
            if self.optimize_3Dpts:
                duv_dxyz = duv_dxyz.reshape(self.npts, self.n_cams * 2, -1)
                jtj_xyz = torch.bmm(duv_dxyz.transpose(1, 2), duv_dxyz)
                jtj_cam_xyz = (
                    torch.bmm(duv_dcam_reshaped.transpose(1, 2), duv_dxyz)
                    .transpose(1, 2)
                    .reshape(-1, self.n_cam_params)
                )

            # Damping
            jtj_cam.diagonal().mul_(1 + lm)
            jtj_cam.diagonal().clamp_min_(self.ep)
            if self.optimize_3Dpts:
                jtj_xyz.diagonal(dim1=-2, dim2=-1).mul_(1 + lm)
                jtj_xyz.diagonal(dim1=-2, dim2=-1).clamp_min_(self.ep)

            # Residual Terms
            jacxr_cam = duv_dcam_flat.T @ r
            if self.optimize_3Dpts:
                jacxr_xyz = torch.bmm(
                    duv_dxyz.transpose(1, 2),
                    r.view(self.npts, self.n_cams * 2).unsqueeze(-1),
                ).view(-1)

                # Solve with Schur Complement
                jtj_xyz_inv = torch.linalg.inv_ex(jtj_xyz)[0]
                jtj_xyz_inv.nan_to_num_()

                # Camera Parameters Update
                BD = (
                    torch.bmm(jtj_xyz_inv, jtj_cam_xyz.view(self.npts, 3, -1))
                    .view(-1, self.n_cam_params)
                    .T
                )
                BmECm1Et = jtj_cam - BD @ jtj_cam_xyz
                BmECm1Et_inv = torch.linalg.inv_ex(BmECm1Et)[0]
                BmECm1Et_inv.nan_to_num_()

                vmECm1w = jacxr_cam - BD @ jacxr_xyz
                dcam = BmECm1Et_inv @ vmECm1w
                dcam.nan_to_num_()

                # 3D Points Update
                b = jacxr_xyz - jtj_cam_xyz @ dcam
                dxyz = torch.bmm(b.view(self.npts, 1, 3), jtj_xyz_inv).view(xyz.shape)
                dxyz.nan_to_num_()
            else:
                # Schur Complement not used when 3D points are not optimized
                dcam = torch.linalg.inv_ex(jtj_cam)[0] @ jacxr_cam
                dcam.nan_to_num_()
                dxyz = 0

            dpose = dcam[: self.n_opt_cams * 9].view(self.n_opt_cams, 9)
            dR = dpose[..., :6].view(-1, 3, 2)
            dt = dpose[..., 6:]
            df = dcam[-1] if self.optimize_focal else 0

            # Check if the update improves residuals. Only the first n_opt_cams are optimized;
            # fixed cameras are kept unchanged in the tail of Rs6D/ts.
            Rs6D_tmp = Rs6D.clone()
            ts_tmp = ts.clone()
            if self.n_opt_cams > 0:
                Rs6D_tmp[: self.n_opt_cams] -= dR
                ts_tmp[: self.n_opt_cams] -= dt
            f_tmp = f.clone() - df
            xyz_tmp = xyz - dxyz

            new_r = self.get_residual(
                *self.prepare_for_proj(xyz_tmp, Rs6D_tmp, ts_tmp, f_tmp, centre), uv
            ).view(-1)
            safe_new_r = torch.nan_to_num(
                new_r, nan=1e6, posinf=1e6, neginf=-1e6
            )
            new_weights = self.get_huber_weights(safe_new_r)
            new_r = torch.where(
                active_coordinates,
                safe_new_r * new_weights,
                torch.zeros_like(safe_new_r),
            )
            new_positive_depth = self.get_positive_depth_mask(
                xyz_tmp, Rs6D_tmp, ts_tmp
            )
            active_observations = active_coordinates.view(
                self.npts, self.n_cams, 2
            ).all(dim=-1)
            n_active_observations = active_observations.sum().clamp_min(1)
            retained_cheirality = (
                new_positive_depth & active_observations
            ).sum() / n_active_observations
            current_positive_count = (
                positive_depth & observation_mask
            ).sum()
            new_positive_count = (
                new_positive_depth & observation_mask
            ).sum()
            keeps_cheirality = (
                (retained_cheirality >= 0.98)
                & (new_positive_count >= current_positive_count)
            )
            success_mask = (
                ((new_r**2).mean() < (r**2).mean())
                * (f_tmp > 0)
                * keeps_cheirality
            )

            # Apply Updates Conditionally
            if self.n_opt_cams > 0:
                Rs6D[: self.n_opt_cams] -= success_mask * dR
                ts[: self.n_opt_cams] -= success_mask * dt
            f = f - success_mask * df
            xyz = xyz - success_mask * dxyz

            # Adjust Damping
            lm *= (1 / self.k) * success_mask + self.k * (1 - success_mask.to(ts))
            Rs6D = mtx2sixD(sixD2mtx(Rs6D))

        # Final residual must correspond to the accepted state.  The previous
        # implementation evaluated the last trial (*_tmp), even when that
        # trial had been rejected, and callers used its mask to accept a
        # different returned pose.
        r = self.get_residual(
            *self.prepare_for_proj(xyz, Rs6D, ts, f, centre), uv
        ).view(-1)
        final_valid = (
            observation_mask
            & self.get_positive_depth_mask(xyz, Rs6D, ts)
            & torch.isfinite(r.view(self.npts, self.n_cams, 2)).all(dim=-1)
        )
        mask = self.get_mask(r, final_valid)
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        initial_r = torch.where(
            torch.isfinite(initial_r), initial_r, torch.zeros_like(initial_r)
        )
        return Rs6D, ts, f, xyz, r, initial_r, mask

    def forward(self, Rs6D, ts, f, xyz, centre, uv):
        if self.batch > 1:
            f = f[None].expand(self.batch, -1)
            centre = centre[None].expand(self.batch, -1)
            return torch.func.vmap(self.optimize)(Rs6D, ts, f, xyz, centre, uv)
        else:
            if Rs6D.ndim == 4:
                Rs6D, ts, xyz, uv = Rs6D[0], ts[0], xyz[0], uv[0]
            return self.optimize(Rs6D, ts, f, xyz, centre, uv)


class MiniBA:
    @torch.no_grad()
    def __init__(
        self,
        batch,
        n_opt_cams,
        n_fixed_cams,
        npts,
        optimize_focal,
        optimize_3Dpts,
        make_cuda_graph=True,
        huber_delta=1,
        outlier_mad_scale=4,
        lm=1e-5,
        ep=1e-2,
        k=2,
        iters=200,
    ):
        """
        Initializes the MiniBA optimizer for camera poses, 3D points and focal length.

        Parameters:
        - batch: number of independant problems
        - n_opt_cams: Number of optimizable cameras.
        - n_fixed_cams: Number of fixed cameras. Fixed cameras are the last in the batch dimension.
        - npts: Number of 3D points.
        - optimize_focal: Optimize focal length if True.
        - optimize_3Dpts: Optimize 3D points if True.
        - make_cuda_graph: Whether to use CUDA graphs for optimization.
        - huber_delta:
        - outlier_mad_scale: If > 0, will not concider points with error > median(error) + outlier_mad_scale * MAD(error)
        - lm: Initial Levenberg-Marquardt damping term.
        - ep: Regularization term to avoid singularities.
        - k: Damping adjustment factor.
        - iters: Number of optimization iterations.
        """

        self.optimizer = MiniBAInternal(
            batch,
            n_opt_cams,
            n_fixed_cams,
            npts,
            optimize_focal,
            optimize_3Dpts,
            huber_delta,
            outlier_mad_scale,
            lm,
            ep,
            k,
            iters,
        )
        self.optimizer = self.optimizer.eval().cuda()

        # Initialize dummy inputs for compilation
        n_cams = n_opt_cams + n_fixed_cams
        Rs6D_init = torch.randn(batch, n_cams, 3, 2, device="cuda")
        ts_init = torch.randn(batch, n_cams, 3, device="cuda")
        f_init = torch.randn(1, device="cuda")
        xyz_init = torch.randn(batch, npts, 3, device="cuda")
        centre = torch.randn(2, device="cuda")
        uv = torch.randn(batch, npts * n_cams * 2, device="cuda")

        if make_cuda_graph:
            self.optimizer = torch.cuda.make_graphed_callables(
                self.optimizer, (Rs6D_init, ts_init, f_init, xyz_init, centre, uv), 10
            )

    @torch.no_grad()
    def __call__(self, Rs6D, ts, f, xyz, centre, uv):
        return self.optimizer(Rs6D, ts, f, xyz, centre, uv)
