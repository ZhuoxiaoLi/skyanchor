#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._n_frozen = 0
        self._freeze_hooks = []

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        xyz_gradient_accum,
        denom,
        opt_dict,
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def freeze_by_mask(self, n_frozen):
        """Freeze the first n_frozen Gaussians by installing zero-gradient hooks."""
        self.unfreeze_all()
        self._n_frozen = n_frozen
        if n_frozen <= 0:
            return

        params = [self._xyz, self._features_dc, self._features_rest,
                  self._opacity, self._scaling, self._rotation]
        for p in params:
            h = p.register_hook(lambda grad, nf=n_frozen: self._zero_frozen_grad(grad, nf))
            self._freeze_hooks.append(h)
        print(f"[Freeze] Frozen first {n_frozen} Gaussians (total: {self.get_xyz.shape[0]})")

    @staticmethod
    def _zero_frozen_grad(grad, n_frozen):
        grad[:n_frozen] = 0
        return grad

    def unfreeze_all(self):
        """Remove all gradient-freezing hooks."""
        for h in self._freeze_hooks:
            h.remove()
        self._freeze_hooks.clear()
        self._n_frozen = 0

    def point_parameters(self, points, colors, init_scale_factor=1.0, init_opacity=0.1):
        """Initialize all point sources using local spacing and a shared opacity."""
        if not math.isfinite(init_scale_factor) or init_scale_factor <= 0:
            raise ValueError("init_scale_factor must be positive and finite")
        if not math.isfinite(init_opacity) or not 0 < init_opacity < 1:
            raise ValueError("init_opacity must be strictly between zero and one")
        xyz = torch.as_tensor(points, dtype=torch.float32, device="cuda").contiguous()
        rgb = torch.as_tensor(colors, dtype=torch.float32, device="cuda")
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4:
            raise ValueError("Gaussian initialization requires at least four 3D points")
        if rgb.shape != xyz.shape or not torch.isfinite(xyz).all() or not torch.isfinite(rgb).all():
            raise ValueError("Initialization points and RGB colors must be finite matching (N, 3) arrays")
        features = torch.zeros((len(xyz), 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, :, 0] = RGB2SH(rgb)
        dist2 = distCUDA2(xyz).clamp_min(1e-7)
        scales = torch.log(torch.sqrt(dist2) * init_scale_factor)[:, None].repeat(1, 3)
        rotations = torch.zeros((len(xyz), 4), device="cuda")
        rotations[:, 0] = 1
        opacities = self.inverse_opacity_activation(torch.full((len(xyz), 1), init_opacity, device="cuda"))
        return (xyz, features[:, :, :1].transpose(1, 2).contiguous(),
                features[:, :, 1:].transpose(1, 2).contiguous(), opacities, scales, rotations)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float,
                        init_scale_factor=1.0, init_opacity=0.1):
        self.spatial_lr_scale = spatial_lr_scale
        values = self.point_parameters(np.asarray(pcd.points), np.asarray(pcd.colors),
                                       init_scale_factor, init_opacity)
        for name, value in zip(("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"), values):
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))
        print(f"[Init] {len(self._xyz)} Gaussians; scale=RMS-3NN * {init_scale_factor:g}; opacity={init_opacity:g}")

    def enforce_frozen_parameters(self):
        """Block both gradients and Adam momentum, including after tensor replacement."""
        if self._n_frozen <= 0:
            return
        for optimizer in (self.optimizer, self.shoptimizer):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad[:self._n_frozen] = 0
                    state = optimizer.state.get(param, {})
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key][:self._n_frozen] = 0

    def optimizer_step(self, visibility=None):
        self.enforce_frozen_parameters()
        if self.optimizer_type == "sparse_adam":
            visibility = visibility.clone()
            visibility[:self._n_frozen] = False
            self.optimizer.step(visibility, len(self._xyz))
        else:
            self.optimizer.step()
            self.shoptimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.shoptimizer is not None:
            self.shoptimizer.zero_grad(set_to_none=True)

    def configure_position_schedule(self, opt, stage_length):
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=opt.position_lr_init * self.spatial_lr_scale,
            lr_final=opt.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult, max_steps=max(int(stage_length), 1))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.shfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, min_opacity):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        if self._n_frozen > 0:
            opacities_new[:self._n_frozen] = self._opacity.data[:self._n_frozen]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask) -> int:
        n = int(mask.sum().item()) if mask is not None and mask.numel() else 0
        if n == 0:
            return 0
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        return n

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def prune_outside_bounds(self, xmin, xmax, ymin, ymax, zmin, zmax, expansion_ratio=0.5) -> int:
        xyz = self.get_xyz
        if xyz.numel() == 0:
            return 0
        dtype = xyz.dtype
        device = xyz.device
        min_bounds = torch.tensor([xmin, ymin, zmin], dtype=dtype, device=device)
        max_bounds = torch.tensor([xmax, ymax, zmax], dtype=dtype, device=device)

        center = (min_bounds + max_bounds) * 0.5
        half_extent = (max_bounds - min_bounds) * 0.5
        expanded_half_extent = half_extent * (1.0 + expansion_ratio)
        expanded_min = center - expanded_half_extent
        expanded_max = center + expanded_half_extent

        outside_mask = ((xyz < expanded_min) | (xyz > expanded_max)).any(dim=1)
        if self._n_frozen > 0:
            outside_mask[:self._n_frozen] = False
        valid_points_mask = ~outside_mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')

        torch.cuda.empty_cache()
        return int(outside_mask.sum().item())

    def densify_and_prune_Improved(self, scores, min_opacity, budget, opt, iteration) -> int:
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0

        min_grad = opt.densify_grad_threshold
        n_frozen = int(getattr(self, "_n_frozen", 0) or 0)
        curr_points = len(self.get_xyz)
        n_train = curr_points - n_frozen
        budget = max(0, int(budget) - n_frozen)

        if scores is None or iteration > 14500:
            scores = grad_vars.squeeze()
            if n_train < budget and iteration > 14500:
                min_grad = min_grad / 1.5

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= min_grad, True, False)
        grad_qualifiers[:n_frozen] = False
        total_sum = torch.sum(grad_qualifiers).item()
        budget = min(budget, total_sum + n_train)
        all_budget = budget - n_train
        n_prune = 0
        if all_budget > 0:
            n_prune += self.long_axis_split(
                scores.clone(), all_budget, grad_qualifiers, opt.split_distance, opt.opacity_reduction
            )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self._n_frozen > 0:
            prune_mask[:self._n_frozen] = False
        n_prune += self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return n_prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter, densification_grad=None):
        grad = viewspace_point_tensor.grad if densification_grad is None else densification_grad
        if grad is None:
            return
        if update_filter.dtype == torch.bool:
            index = update_filter.reshape(-1)
        else:
            index = update_filter.reshape(-1)
        if index.numel() == 0:
            return
        self.xyz_gradient_accum[index] += torch.norm(grad[index,2:], dim=-1, keepdim=True)
        self.denom[index] += 1

    def long_axis_split(self, grads, budget, filter, split_distance, opacity_reduction) -> int:
        grads[~filter] = 0
        if self._n_frozen > 0:
            grads[:self._n_frozen] = 0
        n_init_points = self.get_xyz.shape[0]
        dev = grads.device
        _MAX_MULTINOMIAL = (1 << 24) - 1  # 16777215

        padded_importance = torch.zeros(
            (n_init_points), dtype=torch.float32, device=dev
        )
        padded_importance[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.zeros(
            n_init_points, dtype=torch.bool, device=dev
        )

        pos = torch.where(padded_importance > 0)[0]
        w = padded_importance[pos]
        if w.numel() == 0:
            return 0
        num = w.numel()
        if budget > num:
            budget = num
        if num > _MAX_MULTINOMIAL:
            topw, topi = torch.topk(w, _MAX_MULTINOMIAL, largest=True)
            pos, w = pos[topi], topw
        if budget > w.numel():
            budget = w.numel()
        sampled_idx = torch.multinomial(w, budget, replacement=False)
        selected_pts_mask[pos[sampled_idx]] = True
        stds = self.get_scaling[selected_pts_mask]

        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)

        samples = stds * mask * 3

        reduction = opacity_reduction
        rate = split_distance
        x1 = samples * rate
        rate_w = 1 - rate
        rate_h = math.sqrt(1-rate*rate)
        x1 = torch.cat([x1, -x1], dim=0)

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_xyz = torch.bmm(rots, x1.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_scaling = self.scaling_inverse_activation(stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h)
        new_opacity = inverse_sigmoid(self.get_opacity[selected_pts_mask] * reduction).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        return self.prune_points(prune_filter)
