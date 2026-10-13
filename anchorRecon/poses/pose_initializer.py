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
import math
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from poses.pycolmap_final_ba import run_final_ba as run_pycolmap_final_ba
from poses.triangulator import matches_to_points
from utils import (
    fov2focal,
    depth2points,
    get_global_profiler,
    make_torch_sampler,
    principal_point_tensor,
    sixD2mtx,
    mtx2sixD,
)
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType

class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args, pose_debugger=None):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher
        self.pose_debugger = pose_debugger

        self.centre = principal_point_tensor(
            width,
            height,
            args.init_cx,
            args.init_cy,
            device="cuda",
        )
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_keyframes_miniba_bootstrap = args.num_keyframes_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers
        self.fix_focal = args.fix_focal
        self.iters_miniba_bootstrap = args.iters_miniba_bootstrap
        self.debug_pnp_matching = args.debug_pnp_matching
        self.final_submap_ba_backend = args.final_submap_ba_backend
        self.final_submap_ba_pycolmap_solver = args.final_submap_ba_pycolmap_solver
        self.final_submap_ba_min_track_obs = args.final_submap_ba_min_track_obs
        self.final_submap_ba_anchor_match_frames = int(
            getattr(args, "final_submap_ba_anchor_match_frames", 8)
        )
        self.final_submap_ba_max_depth = args.final_submap_ba_max_depth
        self.pnp_max_depth = float(getattr(args, "pnp_max_depth", -1.0))
        self.bootstrap_seed_init_mode = getattr(args, "bootstrap_seed_init_mode", "anchor_pnp")
        self.skip_incremental_magsac = bool(
            getattr(args, "skip_incremental_magsac", False)
        )
        if self.skip_incremental_magsac:
            print(
                "[pose_initializer] --skip_incremental_magsac=True: incremental matcher loop "
                "will rely on PnP-RANSAC for outlier rejection (no MAGSAC fundamental-matrix filter)."
            )

        # Initialize the focal length
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
                      prefer_long_tracks: bool = False,
                      return_track_meta: bool = False,
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        track_primary = torch.full((npts,), -1, dtype=torch.int64, device="cuda")
        track_score_sum = torch.zeros(npts, dtype=torch.float32, device="cuda")
        track_score_count = torch.zeros(npts, dtype=torch.float32, device="cuda")
        n_keypoints = desc_kpts_list[0].kpts.shape[0]
        # A 2D observation may belong to exactly one 3D track.  The previous
        # implementation only marked direct k->l assignments as used; a
        # transitive l->m assignment could therefore be selected again as the
        # primary of a second track.  Besides double-counting residuals, that
        # produces conflicting COLMAP point2D->point3D associations and weakens
        # the Schur system.  Keep the ownership map explicit for every camera.
        keypoint_track_ids = torch.full(
            (n_cams, n_keypoints),
            -1,
            dtype=torch.int64,
            device="cuda",
        )

        def claim_pair_observations(
            source_cam: int,
            target_cam: int,
            pair_matches,
            track_begin: int,
            track_end: int,
        ):
            """Attach pairwise matches without duplicating a camera observation."""
            source_idx = pair_matches.idx
            target_idx = pair_matches.idx_other
            source_track = keypoint_track_ids[source_cam, source_idx]
            valid = (source_track >= track_begin) & (source_track < track_end)
            if not valid.any():
                return

            source_track = source_track[valid]
            target_idx = target_idx[valid]
            if pair_matches.scores is None:
                scores = torch.zeros(
                    len(source_track), dtype=torch.float32, device=uvs.device
                )
            else:
                scores = pair_matches.scores[valid].float()

            target_owner = keypoint_track_ids[target_cam, target_idx]
            row_is_empty = xyz_indices[source_track, target_cam] < 0
            compatible = row_is_empty & (
                (target_owner < 0) | (target_owner == source_track)
            )
            if not compatible.any():
                return

            source_track = source_track[compatible]
            target_idx = target_idx[compatible]
            scores = scores[compatible]

            # Pairwise matchers are normally one-to-one, but enforce it here
            # as well.  When several candidates claim the same target keypoint
            # or the same camera slot, retain the highest-confidence edge.
            order = torch.argsort(scores, descending=True, stable=True)
            source_track = source_track[order]
            target_idx = target_idx[order]
            scores = scores[order]
            rank = torch.arange(len(order), device=uvs.device, dtype=torch.int64)

            first_for_target = torch.full(
                (n_keypoints,), len(order), device=uvs.device, dtype=torch.int64
            )
            first_for_target.scatter_reduce_(
                0, target_idx, rank, reduce="amin", include_self=True
            )
            keep = rank == first_for_target[target_idx]
            source_track = source_track[keep]
            target_idx = target_idx[keep]
            scores = scores[keep]
            rank = torch.arange(
                len(source_track), device=uvs.device, dtype=torch.int64
            )

            first_for_track = torch.full(
                (npts,), len(source_track), device=uvs.device, dtype=torch.int64
            )
            first_for_track.scatter_reduce_(
                0, source_track, rank, reduce="amin", include_self=True
            )
            keep = rank == first_for_track[source_track]
            source_track = source_track[keep]
            target_idx = target_idx[keep]
            scores = scores[keep]
            if len(source_track) == 0:
                return

            uvs[source_track, target_cam] = desc_kpts_list[target_cam].kpts[
                target_idx
            ]
            xyz_indices[source_track, target_cam] = target_idx
            keypoint_track_ids[target_cam, target_idx] = source_track
            track_score_sum.index_add_(0, source_track, scores)
            track_score_count.index_add_(
                0, source_track, torch.ones_like(scores)
            )

        bucket_sizes = [
            npts // n_primary_cam + int(k < (npts % n_primary_cam))
            for k in range(n_primary_cam)
        ]
        bucket_start = 0
        for k in range(n_primary_cam):
            npts_per_primary_cam = bucket_sizes[k]
            bucket_end = bucket_start + npts_per_primary_cam
            track_primary[bucket_start:bucket_end] = k
            idx_occurrences = torch.zeros(n_keypoints, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= keypoint_track_ids[k] < 0
            if idx_occurrences.sum() == 0:
                print("No matches.")
                bucket_start = bucket_end
                continue
            valid_occurrences = idx_occurrences > 0
            n_selected = min(int(valid_occurrences.sum().item()), npts_per_primary_cam)
            if n_selected == 0:
                bucket_start = bucket_end
                continue
            sampling_weights = (
                idx_occurrences.float()
                if prefer_long_tracks
                else valid_occurrences.float()
            )
            selected_indices = torch.multinomial(
                sampling_weights, n_selected, replacement=False
            )

            aligned_ids = torch.arange(bucket_start, bucket_start + n_selected, device="cuda")
            uvs[aligned_ids, k] = desc_kpts_list[k].kpts[selected_indices]
            xyz_indices[aligned_ids, k] = selected_indices
            keypoint_track_ids[k, selected_indices] = aligned_ids

            for l in range(n_cams):
                if l == k:
                    continue
                lId = kfId_list[l]
                if lId not in desc_kpts_list[k].matches:
                    continue
                claim_pair_observations(
                    k,
                    l,
                    desc_kpts_list[k].matches[lId],
                    bucket_start,
                    bucket_start + n_selected,
                )

                # Extend the just-created tracks by one more pairwise edge.
                # Ownership checks above make this propagation cycle-safe: a
                # conflicting target is rejected instead of being copied into
                # two 3D tracks.
                for m in range(l + 1, n_cams):
                    if m == k:
                        continue
                    mId = kfId_list[m]
                    if mId in desc_kpts_list[l].matches:
                        claim_pair_observations(
                            l,
                            m,
                            desc_kpts_list[l].matches[mId],
                            bucket_start,
                            bucket_start + n_selected,
                        )
            bucket_start = bucket_end

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        if not return_track_meta:
            return uvs, xyz_indices

        track_quality = track_score_sum / track_score_count.clamp_min(1.0)
        track_quality = torch.where(
            track_score_count > 0,
            track_quality,
            torch.zeros_like(track_quality),
        )
        return uvs, xyz_indices, {
            "track_primary": track_primary,
            "track_quality": track_quality,
            "track_score_count": track_score_count,
        }

    @torch.no_grad()
    def select_final_ba_tracks(
        self,
        uvs,
        track_quality: torch.Tensor,
        track_primary: torch.Tensor,
        npts: int,
        n_movable_cams: int,
        min_track_obs: int = 3,
    ):
        obs_mask = (uvs >= 0).all(dim=-1)
        n_obs = obs_mask.sum(dim=-1)
        n_movable_obs = obs_mask[:, :n_movable_cams].sum(dim=-1)
        n_fixed_obs = obs_mask[:, n_movable_cams:].sum(dim=-1)
        anchor_connected = (n_movable_obs > 0) & (n_fixed_obs > 0)
        valid = n_obs >= 2
        preferred = n_obs >= min_track_obs
        primary_valid = (track_primary >= 0) & (track_primary < n_movable_cams)

        score = track_quality.float().clone()
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        score += 0.05 * n_obs.float()

        selected_mask = torch.zeros(uvs.shape[0], dtype=torch.bool, device=uvs.device)
        selected_chunks = []
        bucket_sizes = [
            npts // n_movable_cams + int(k < (npts % n_movable_cams))
            for k in range(n_movable_cams)
        ]
        def choose_from(mask, budget):
            if budget <= 0:
                return None
            ids = torch.where(mask & ~selected_mask)[0]
            if len(ids) == 0:
                return None
            order = torch.argsort(score[ids], descending=True, stable=True)
            return ids[order[: min(budget, len(ids))]]

        for primary_id, bucket_size in enumerate(bucket_sizes):
            if bucket_size <= 0:
                continue
            primary_mask = primary_valid & (track_primary == primary_id)
            remaining = bucket_size
            # Boundary constraints must not lose a popularity contest against
            # a much larger internally-consistent local chain.  Within every
            # movable-camera bucket, take fixed-anchor-connected tracks first.
            for category in (
                valid & preferred & anchor_connected & primary_mask,
                valid & anchor_connected & primary_mask,
                valid & preferred & primary_mask,
                valid & primary_mask,
            ):
                chosen = choose_from(category, remaining)
                if chosen is None:
                    continue
                selected_mask[chosen] = True
                selected_chunks.append(chosen)
                remaining -= len(chosen)
                if remaining <= 0:
                    break

        n_selected = sum(len(chunk) for chunk in selected_chunks)
        if n_selected < npts:
            for category in (
                valid & preferred & anchor_connected & primary_valid,
                valid & anchor_connected & primary_valid,
                valid & preferred & primary_valid,
                valid & primary_valid,
            ):
                chosen = choose_from(category, npts - n_selected)
                if chosen is None:
                    continue
                selected_mask[chosen] = True
                selected_chunks.append(chosen)
                n_selected += len(chosen)
                if n_selected >= npts:
                    break

        if n_selected < npts:
            remainder_ids = torch.where(~selected_mask)[0]
            remainder_score = score[remainder_ids] + valid[remainder_ids].float() * (uvs.shape[1] + 1)
            remainder_order = torch.argsort(remainder_score, descending=True)
            chosen = remainder_ids[remainder_order[: min(npts - n_selected, len(remainder_ids))]]
            selected_chunks.append(chosen)

        if selected_chunks:
            selected_ids = torch.cat(selected_chunks, dim=0)[:npts]
        else:
            selected_ids = torch.arange(
                min(npts, uvs.shape[0]), device=uvs.device
            )
        selected_uvs = uvs[selected_ids]

        selected_obs_mask = obs_mask[selected_ids]
        selected_n_obs = selected_obs_mask.sum(dim=-1)
        selected_anchor_connected = (
            selected_obs_mask[:, :n_movable_cams].any(dim=-1)
            & selected_obs_mask[:, n_movable_cams:].any(dim=-1)
        )
        selected_valid = selected_n_obs >= 2
        selected_preferred = selected_n_obs >= min_track_obs
        selected_quality = track_quality[selected_ids]
        selected_non_anchor = selected_valid & ~selected_anchor_connected
        selected_valid_obs = selected_n_obs[selected_valid]
        obs_distribution = []
        for obs_count in [2, 3, 4]:
            obs_distribution.append(
                f"{obs_count}:{int((selected_valid_obs == obs_count).sum().item())}"
            )
        obs_distribution.append(
            f"5+:{int((selected_valid_obs >= 5).sum().item())}"
        )
        stats = {
            "candidate_tracks": int(valid.sum().item()),
            "candidate_anchor_tracks": int((valid & anchor_connected).sum().item()),
            "candidate_non_anchor_tracks": int((valid & ~anchor_connected).sum().item()),
            "selected_tracks": int(selected_valid.sum().item()),
            "selected_anchor_tracks": int(
                (selected_valid & selected_anchor_connected).sum().item()
            ),
            "selected_non_anchor_tracks": int(selected_non_anchor.sum().item()),
            "selected_preferred_tracks": int((selected_valid & selected_preferred).sum().item()),
            "selected_mean_obs": float(
                selected_n_obs[selected_valid].float().mean().item()
                if selected_valid.any()
                else 0.0
            ),
            "selected_mean_score": float(
                selected_quality[selected_valid].float().mean().item()
                if selected_valid.any()
                else 0.0
            ),
            "selected_obs_distribution": ",".join(obs_distribution),
        }
        selection = {
            "candidate_valid": valid,
            "candidate_anchor_connected": anchor_connected,
            "selected_ids": selected_ids,
            "selected_valid": selected_valid,
            "selected_anchor_connected": selected_anchor_connected,
            "selected_preferred": selected_preferred,
        }
        return selected_uvs, stats, selection

    @torch.no_grad()
    def default_xyz_init(self, uvs, f):
        xyz_init = torch.zeros(uvs.shape[0], 3, device="cuda")
        for k in range(uvs.shape[1]):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()
        return xyz_init

    @torch.no_grad()
    def triangulate_xyz_init(self, uvs, Rts, f, debug_label: str = ""):
        xyz_init = self.default_xyz_init(uvs, f)
        valid_tracks = torch.zeros(uvs.shape[0], dtype=torch.bool, device="cuda")
        Rts_inv = torch.linalg.inv_ex(Rts)[0]
        debug_counts = {
            "tested": 0,
            "finite": 0,
            "positive_depth": 0,
            "reprojection": 0,
            "parallax": 0,
            "all": 0,
        }
        debug_errors = []
        debug_disambiguations = []
        debug_baselines = []

        # The old implementation launched several tiny CUDA kernels per track
        # from Python (10k--30k tracks), which dominated final-BA runtime.  A
        # track's first observed camera is still used as its reference, but all
        # tracks sharing a reference/other camera pair are triangulated in one
        # batch.  We also enforce cheirality in both views; checking only the
        # reference view admitted flipped/behind-camera bootstrap solutions.
        obs_mask = (uvs >= 0).all(dim=-1)
        n_obs = obs_mask.sum(dim=-1)
        first_observer = obs_mask.to(torch.int64).argmax(dim=-1)
        best_disambiguation = torch.full(
            (uvs.shape[0],), -1.0, device=uvs.device, dtype=uvs.dtype
        )

        for ref_id in range(uvs.shape[1]):
            ref_track_ids = torch.where(
                (n_obs >= 2) & (first_observer == ref_id)
            )[0]
            if len(ref_track_ids) == 0:
                continue
            for other_id in range(ref_id + 1, uvs.shape[1]):
                track_ids = ref_track_ids[
                    obs_mask[ref_track_ids, other_id]
                ]
                if len(track_ids) == 0:
                    continue

                rel_Rt = Rts[ref_id] @ Rts_inv[other_id]
                xyz_cam, disambiguation, error = matches_to_points(
                    uvs[track_ids, ref_id],
                    uvs[track_ids, other_id],
                    rel_Rt[:3, :3],
                    rel_Rt[:3, 3],
                    f,
                    self.centre,
                )
                xyz_other_cam = (
                    xyz_cam - rel_Rt[None, :3, 3]
                ) @ rel_Rt[:3, :3]
                finite = (
                    torch.isfinite(xyz_cam).all(dim=-1)
                    & torch.isfinite(xyz_other_cam).all(dim=-1)
                    & torch.isfinite(disambiguation)
                    & torch.isfinite(error)
                )
                positive_depth = (
                    finite
                    & (xyz_cam[:, 2] > 1e-6)
                    & (xyz_other_cam[:, 2] > 1e-6)
                )
                reprojection_valid = finite & (error < self.max_pnp_error)
                parallax_valid = (
                    finite
                    # The online triangulator uses match_error*30.  PnP's
                    # threshold is twice match_error, hence the factor 15.
                    & (disambiguation > self.max_pnp_error * 15.0)
                )
                pair_valid = (
                    positive_depth
                    & reprojection_valid
                    & parallax_valid
                    & (disambiguation > best_disambiguation[track_ids])
                )
                if debug_label:
                    debug_counts["tested"] += len(track_ids)
                    debug_counts["finite"] += int(finite.sum().item())
                    debug_counts["positive_depth"] += int(positive_depth.sum().item())
                    debug_counts["reprojection"] += int(reprojection_valid.sum().item())
                    debug_counts["parallax"] += int(parallax_valid.sum().item())
                    debug_counts["all"] += int(
                        (positive_depth & reprojection_valid & parallax_valid).sum().item()
                    )
                    if finite.any():
                        debug_errors.append(error[finite])
                        debug_disambiguations.append(disambiguation[finite])
                    debug_baselines.append(float(rel_Rt[:3, 3].norm().item()))
                if not pair_valid.any():
                    continue
                update_ids = track_ids[pair_valid]
                xyz_world = (
                    xyz_cam[pair_valid] - Rts[ref_id, None, :3, 3]
                ) @ Rts[ref_id, :3, :3]
                xyz_init[update_ids] = xyz_world
                best_disambiguation[update_ids] = disambiguation[pair_valid]
                valid_tracks[update_ids] = True

        if debug_label:
            def quantiles(values):
                if not values:
                    return []
                joined = torch.cat(values).float()
                return [
                    round(float(value), 4)
                    for value in torch.quantile(
                        joined,
                        torch.tensor(
                            [0.0, 0.25, 0.5, 0.75, 1.0],
                            device=joined.device,
                        ),
                    ).tolist()
                ]

            baseline_range = (
                [round(min(debug_baselines), 6), round(max(debug_baselines), 6)]
                if debug_baselines
                else []
            )
            print(
                f"[{debug_label}] counts={debug_counts}, "
                f"error_q0/25/50/75/100={quantiles(debug_errors)}, "
                f"disambiguation_q0/25/50/75/100={quantiles(debug_disambiguations)}, "
                f"baseline_range={baseline_range}, "
                f"thresholds(error<{self.max_pnp_error:.4f}, "
                f"disambiguation>{self.max_pnp_error * 15.0:.4f})"
            )

        return xyz_init, valid_tracks

    @torch.no_grad()
    def filter_tracks_by_depth(
        self,
        uvs: torch.Tensor,
        xyz: torch.Tensor,
        valid_tracks: torch.Tensor,
        Rts: torch.Tensor,
        max_depth: float,
        n_movable_cams: int | None = None,
    ):
        filtered_valid = valid_tracks.clone()
        if not valid_tracks.any():
            return filtered_valid, {
                "selected_depth_kept_tracks": 0,
                "selected_depth_filtered_tracks": 0,
            }

        valid_ids = torch.where(valid_tracks)[0]
        valid_uvs = uvs[valid_ids]
        valid_xyz = xyz[valid_ids]
        obs_mask = (valid_uvs >= 0).all(dim=-1)

        xyz_cam = torch.einsum("cij,pj->cpi", Rts[:, :3, :3], valid_xyz)
        xyz_cam = xyz_cam + Rts[:, None, :3, 3]
        depths = xyz_cam[..., 2].transpose(0, 1)

        # A track observed by a fixed anchor but lying behind that anchor is
        # geometrically incompatible with the boundary condition and must not
        # enter BA.  This check is always active, independently of max_depth.
        if n_movable_cams is not None and n_movable_cams < uvs.shape[1]:
            fixed_obs = obs_mask[:, n_movable_cams:]
            fixed_depths = depths[:, n_movable_cams:]
            fixed_cheirality = (
                ~fixed_obs
                | (torch.isfinite(fixed_depths) & (fixed_depths > 1e-6))
            ).all(dim=-1)
        else:
            fixed_cheirality = torch.ones(
                len(valid_ids), dtype=torch.bool, device=uvs.device
            )

        if max_depth is None or max_depth <= 0:
            filtered_valid[valid_ids] = fixed_cheirality
            return filtered_valid, {
                "selected_depth_kept_tracks": int(fixed_cheirality.sum().item()),
                "selected_depth_filtered_tracks": int((~fixed_cheirality).sum().item()),
            }

        positive_depths = torch.where(
            obs_mask & (depths > 0),
            depths,
            torch.full_like(depths, float("inf")),
        )
        closest_positive_depth = positive_depths.min(dim=-1).values
        keep_valid = torch.isfinite(closest_positive_depth) & (
            closest_positive_depth <= float(max_depth)
        ) & fixed_cheirality
        filtered_valid[valid_ids] = keep_valid
        return filtered_valid, {
            "selected_depth_kept_tracks": int(keep_valid.sum().item()),
            "selected_depth_filtered_tracks": int((~keep_valid).sum().item()),
        }

    @torch.no_grad()
    def sample_track_colors(self, keyframes: list[Keyframe], uvs):
        colors = torch.zeros((uvs.shape[0], 3), device="cuda", dtype=torch.float32)
        counts = torch.zeros((uvs.shape[0], 1), device="cuda", dtype=torch.float32)

        for cam_idx, keyframe in enumerate(keyframes):
            valid = (uvs[:, cam_idx, :] >= 0).all(dim=-1)
            if not valid.any():
                continue

            image = keyframe.image_pyr[0]
            if image.device.type != "cuda":
                image = image.cuda()
            uv = uvs[valid, cam_idx, :]
            sampler = make_torch_sampler(uv.view(1, 1, -1, 2), keyframe.width, keyframe.height)
            sampled = F.grid_sample(
                image[None],
                sampler,
                mode="bilinear",
                align_corners=True,
            )[0, :, 0].transpose(0, 1)
            colors[valid] += sampled
            counts[valid] += 1

        colors = colors / counts.clamp_min(1.0)
        return (colors.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)



    @torch.no_grad()
    def initialize_bootstrap_seeded(
        self,
        desc_kpts_list: list[DescribedKeypoints],
        seed_Rts: torch.Tensor,
    ):
        """
        Estimate focal and the non-seed bootstrap poses while keeping seed poses fixed.
        """
        n_cams = len(desc_kpts_list)
        n_fixed = seed_Rts.shape[0]
        n_opt = n_cams - n_fixed
        npts = self.num_pts_miniba_bootstrap
        if n_fixed <= 0 or n_fixed > n_cams:
            raise ValueError("seed_Rts must contain between 1 and n_cams poses.")

        f_init = getattr(
            self, "f", torch.tensor([self.f_init], device="cuda")
        ).clone()
        if n_opt == 0:
            self.f = f_init
            self.intrinsics = torch.cat([f_init, self.centre], dim=0)
            return seed_Rts.clone(), f_init, torch.zeros((), device="cuda")

        if n_opt > 0:
            mode = getattr(self, "bootstrap_seed_init_mode", "anchor_pnp")
            if not getattr(self, "_seeded_bootstrap_mode_logged", False):
                seed_centers = torch.linalg.inv(seed_Rts)[:, :3, 3]
                seed_spread = float((seed_centers - seed_centers.mean(0)).norm(dim=-1).max())
                print(
                    f"[bootstrap] seeded init mode='{mode}' "
                    f"(n_fixed={n_fixed}, n_opt={n_opt}, seed_spread_max={seed_spread:.4f})"
                )
                self._seeded_bootstrap_mode_logged = True
            bootstrap_pair_stats = []
            with get_global_profiler().stage("bootstrap.pairwise_matching"):
                for i in range(n_cams):
                    for j in range(i + 1, n_cams):
                        _ = self.matcher(
                            desc_kpts_list[i],
                            desc_kpts_list[j],
                            remove_outliers=True,
                            update_kpts_flag="inliers",
                            kID=i,
                            kID_other=j,
                        )
                        if self.debug_pnp_matching:
                            stats = self.matcher.last_match_stats
                            bootstrap_pair_stats.append(
                                (
                                    i,
                                    j,
                                    int(stats.get("raw_matches", 0)),
                                    int(stats.get("geom_inliers", 0)),
                                )
                            )
            if self.debug_pnp_matching:
                for free_id in range(n_fixed, n_cams):
                    anchor_pairs = [
                        (i, raw, geom)
                        for i, j, raw, geom in bootstrap_pair_stats
                        if j == free_id and i < n_fixed
                    ]
                    print(
                        f"[bootstrap/matches] free={free_id - n_fixed}, "
                        f"anchor_pairs(anchor,raw,geom)={anchor_pairs}"
                    )
                raw_total = sum(item[2] for item in bootstrap_pair_stats)
                geom_total = sum(item[3] for item in bootstrap_pair_stats)
                print(
                    f"[bootstrap/matches] pairs={len(bootstrap_pair_stats)}, "
                    f"raw_total={raw_total}, geom_total={geom_total}"
                )
            Rts_init = torch.eye(
                4, device=seed_Rts.device, dtype=seed_Rts.dtype
            )[None].repeat(n_cams, 1, 1)
            Rts_init[:n_fixed] = seed_Rts
            if mode == "seed_init" or n_fixed < 2:
                Rts_init[n_fixed:] = seed_Rts[-1:].repeat(n_opt, 1, 1)
            else:
                # The seed anchors are temporally consecutive.  Constant-
                # velocity extrapolation gives PnP and the joint BA a
                # physically meaningful basin without changing the 6+2
                # anchored initialization design.
                seed_c2w = torch.linalg.inv(seed_Rts)
                delta_c2w = torch.linalg.inv(seed_c2w[-2]) @ seed_c2w[-1]
                current_c2w = seed_c2w[-1]
                for free_offset in range(n_opt):
                    current_c2w = current_c2w @ delta_c2w
                    Rts_init[n_fixed + free_offset] = torch.linalg.inv(
                        current_c2w
                    )

        opt_ids = list(range(n_fixed, n_cams))
        fixed_ids = list(range(n_fixed))
        ordered_ids = opt_ids + fixed_ids
        ordered_desc_kpts = [desc_kpts_list[i] for i in ordered_ids]
        ordered_Rts_init = Rts_init[ordered_ids]
        with get_global_profiler().stage("bootstrap.build_problem"):
            uvs, _ = self.build_problem(
                ordered_desc_kpts,
                npts,
                n_cams,
                n_opt,
                2,
                ordered_ids,
                prefer_long_tracks=True,
            )

        # Every retained bootstrap point must be observed by a free frame and
        # triangulatable from at least two fixed anchors.  This removes both
        # fixed-only tracks (no free-pose information) and free-only tracks
        # (an independent Sim(3) gauge).
        obs_mask = (uvs >= 0).all(dim=-1)
        fixed_obs_count = obs_mask[:, n_opt:].sum(dim=-1)
        bootstrap_connected = (
            obs_mask[:, :n_opt].any(dim=-1) & (fixed_obs_count >= 2)
        )
        anchor_uvs = uvs[:, n_opt:].clone()
        with get_global_profiler().stage("bootstrap.triangulate_init"):
            xyz_init, anchor_valid = self.triangulate_xyz_init(
                anchor_uvs,
                ordered_Rts_init[n_opt:],
                f_init,
                debug_label=(
                    "bootstrap/triangulation" if self.debug_pnp_matching else ""
                ),
            )
        bootstrap_valid = bootstrap_connected & anchor_valid
        uvs[~bootstrap_valid] = -1

        if self.debug_pnp_matching:
            fixed_hist = [
                int((fixed_obs_count == count).sum().item())
                for count in range(n_fixed + 1)
            ]
            per_free_connected = [
                int(
                    (
                        obs_mask[:, free_id]
                        & (fixed_obs_count >= 2)
                    ).sum().item()
                )
                for free_id in range(n_opt)
            ]
            per_free_triangulated = [
                int(
                    (
                        obs_mask[:, free_id]
                        & (fixed_obs_count >= 2)
                        & anchor_valid
                    ).sum().item()
                )
                for free_id in range(n_opt)
            ]
            print(
                "[bootstrap/graph] "
                f"fixed_obs_hist={fixed_hist}, "
                f"connected={int(bootstrap_connected.sum().item())}, "
                f"anchor_triangulated={int(anchor_valid.sum().item())}, "
                f"valid={int(bootstrap_valid.sum().item())}, "
                f"per_free_connected={per_free_connected}, "
                f"per_free_valid={per_free_triangulated}"
            )

        required_tracks = max(20, self.min_num_inliers)
        per_free_tracks = []
        pnp_inliers_per_free = []
        for free_id in range(n_opt):
            free_track_mask = bootstrap_valid & obs_mask[:, free_id]
            n_free_tracks = int(free_track_mask.sum().item())
            per_free_tracks.append(n_free_tracks)
            if n_free_tracks < required_tracks:
                raise RuntimeError(
                    "Seeded bootstrap is underconstrained: "
                    f"free camera {free_id} has {n_free_tracks} fixed-anchor "
                    f"3D correspondences, need at least {required_tracks}."
                )

            if mode == "anchor_pnp":
                pnp_uv = uvs[free_track_mask, free_id]
                pnp_xyz = xyz_init[free_track_mask]
                pnp_conf = torch.ones(
                    n_free_tracks, device=pnp_uv.device, dtype=pnp_uv.dtype
                )
                with get_global_profiler().stage("bootstrap.anchor_pnp"):
                    pnp_Rt, pnp_inliers = self.PnPRANSAC(
                        pnp_uv,
                        pnp_xyz,
                        f_init,
                        self.centre,
                        mtx2sixD(ordered_Rts_init[free_id, :3, :3]),
                        ordered_Rts_init[free_id, :3, 3],
                        pnp_conf,
                    )
                n_pnp_inliers = int(pnp_inliers.sum().item())
                pnp_inliers_per_free.append(n_pnp_inliers)
                if n_pnp_inliers < required_tracks:
                    raise RuntimeError(
                        "Seeded bootstrap anchor-PnP failed: "
                        f"free camera {free_id} has {n_pnp_inliers}/{n_free_tracks} "
                        f"inliers, need at least {required_tracks}."
                    )
                free_track_ids = torch.where(free_track_mask)[0]
                uvs[free_track_ids[~pnp_inliers], free_id] = -1
                ordered_Rts_init[free_id, :3, :4] = pnp_Rt
            else:
                pnp_inliers_per_free.append(-1)

        # PnP rejection may remove the only free observation of a track.
        # Do not let such a newly fixed-only row consume MiniBA capacity.
        joint_valid = bootstrap_valid & (uvs[:, :n_opt] >= 0).all(
            dim=-1
        ).any(dim=-1)
        uvs[~joint_valid] = -1

        print(
            "[bootstrap] fixed-connected tracks="
            f"{int(joint_valid.sum().item())}/{npts}, "
            f"per_free={per_free_tracks}, anchor_pnp_inliers={pnp_inliers_per_free}"
        )

        miniba_seeded = MiniBA(
            1,
            n_opt,
            n_fixed,
            npts,
            optimize_focal=not self.fix_focal,
            optimize_3Dpts=True,
            make_cuda_graph=False,
            iters=self.iters_miniba_bootstrap,
        )
        Rs6D_init = mtx2sixD(ordered_Rts_init[:, :3, :3])
        ts_init = ordered_Rts_init[:, :3, 3]
        with get_global_profiler().stage("bootstrap.miniba"):
            Rs6D, ts, f, xyz, r, r_init, mask = miniba_seeded(
                Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1)
            )
        final_residual = (r * mask).abs().sum() / mask.sum().clamp_min(1)

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ordered_Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        ordered_Rts[:, :3, :3] = sixD2mtx(Rs6D)
        ordered_Rts[:, :3, 3] = ts

        Rts = torch.empty_like(ordered_Rts)
        for ordered_idx, original_idx in enumerate(ordered_ids):
            Rts[original_idx] = ordered_Rts[ordered_idx]
        Rts[:n_fixed] = seed_Rts

        return Rts, f, final_residual

    @torch.no_grad()
    def finalize_submap_ba(
        self,
        keyframes: list[Keyframe],
        npts: int,
        iters: int,
        optimize_focal: bool = False,
        candidate_multiplier: float = 1.0,
        return_point_colors: bool = False,
        return_track_groups: bool = False,
        return_colmap_payload: bool = False,
        require_tail_connectivity: bool = True,
        strict_graph: bool = True,
        causal_front_only: bool = False,
    ):
        """
        Refine the submap poses while keeping all frames that have anchor_Rt fixed.
        """
        if causal_front_only:
            leaked_tail = [
                keyframe.info.get("name", keyframe.index)
                for keyframe in keyframes
                if keyframe.info.get("is_submap_tail_anchor", False)
            ]
            leaked_non_front_anchor = [
                keyframe.info.get("name", keyframe.index)
                for keyframe in keyframes
                if keyframe.info.get("anchor_Rt", None) is not None
                and not keyframe.info.get("is_submap_seed_anchor", False)
            ]
            if leaked_tail or leaked_non_front_anchor:
                raise RuntimeError(
                    "Causal front-anchor BA was given future/non-front anchors: "
                    f"tail={leaked_tail}, non_front_fixed={leaked_non_front_anchor}. "
                    "Refusing to leak back-boundary poses into online localization."
                )
            if require_tail_connectivity:
                raise ValueError(
                    "causal_front_only=True is incompatible with "
                    "require_tail_connectivity=True."
                )

        movable_keyframes = []
        fixed_keyframes = []
        for keyframe in keyframes:
            if keyframe.info.get("anchor_Rt", None) is not None:
                fixed_keyframes.append(keyframe)
            else:
                movable_keyframes.append(keyframe)

        if len(fixed_keyframes) == 0 or len(movable_keyframes) == 0:
            return None

        tail_anchors = [
            keyframe
            for keyframe in fixed_keyframes
            if keyframe.info.get("is_submap_tail_anchor", False)
        ]
        n_anchor_match_frames = min(
            max(self.final_submap_ba_anchor_match_frames, 0),
            len(movable_keyframes),
        )
        n_new_anchor_pairs = 0
        if tail_anchors and n_anchor_match_frames > 0:
            # Online registration only matches four selected predecessors.
            # Explicitly close the graph at the back boundary before BA; merely
            # increasing the candidate-track budget cannot create constraints
            # that were never matched.
            tail_movable = movable_keyframes[-n_anchor_match_frames:]
            with get_global_profiler().stage("final_ba.anchor_matching"):
                for anchor in tail_anchors:
                    for movable in tail_movable:
                        if anchor.index in movable.desc_kpts.matches:
                            continue
                        self.matcher(
                            movable.desc_kpts,
                            anchor.desc_kpts,
                            remove_outliers=True,
                            update_kpts_flag="inliers",
                            kID=movable.index,
                            kID_other=anchor.index,
                        )
                        n_new_anchor_pairs += 1
            print(
                "[final_ba] explicit tail loop matching: "
                f"anchors={len(tail_anchors)}, movable={n_anchor_match_frames}, "
                f"new_pairs={n_new_anchor_pairs}"
            )

        ordered_keyframes = movable_keyframes + fixed_keyframes
        desc_kpts_list = [keyframe.desc_kpts for keyframe in ordered_keyframes]
        kfId_list = [keyframe.index for keyframe in ordered_keyframes]
        n_cams = len(ordered_keyframes)
        candidate_npts = max(npts, int(round(npts * max(candidate_multiplier, 1.0))))
        candidate_uvs, _, candidate_meta = self.build_problem(
            desc_kpts_list,
            candidate_npts,
            n_cams,
            len(movable_keyframes),
            2,
            kfId_list,
            prefer_long_tracks=True,
            return_track_meta=True,
        )
        selected_uvs, track_stats, selection = self.select_final_ba_tracks(
            candidate_uvs,
            candidate_meta["track_quality"],
            candidate_meta["track_primary"],
            npts,
            len(movable_keyframes),
            min_track_obs=self.final_submap_ba_min_track_obs,
        )

        Rts_init = torch.stack(
            [
                keyframe.get_Rt()
                if keyframe.info.get("anchor_Rt", None) is None
                else keyframe.info["anchor_Rt"]
                for keyframe in ordered_keyframes
            ],
            dim=0,
        )
        f_init = self.f.clone()
        xyz_init, valid_tracks = self.triangulate_xyz_init(selected_uvs, Rts_init, f_init)
        valid_tracks, depth_stats = self.filter_tracks_by_depth(
            selected_uvs,
            xyz_init,
            valid_tracks,
            Rts_init,
            self.final_submap_ba_max_depth,
            n_movable_cams=len(movable_keyframes),
        )
        selected_track_quality = candidate_meta["track_quality"][selection["selected_ids"]]
        if valid_tracks.any():
            valid_obs_mask = (selected_uvs[valid_tracks] >= 0).all(dim=-1)
            valid_n_obs = valid_obs_mask.sum(dim=-1)
            valid_anchor = selection["selected_anchor_connected"][valid_tracks]
            valid_preferred = selection["selected_preferred"][valid_tracks]
            obs_distribution = []
            for obs_count in [2, 3, 4]:
                obs_distribution.append(
                    f"{obs_count}:{int((valid_n_obs == obs_count).sum().item())}"
                )
            obs_distribution.append(f"5+:{int((valid_n_obs >= 5).sum().item())}")
            track_stats.update(
                {
                    "selected_tracks": int(valid_tracks.sum().item()),
                    "selected_anchor_tracks": int(valid_anchor.sum().item()),
                    "selected_non_anchor_tracks": int((~valid_anchor).sum().item()),
                    "selected_preferred_tracks": int(valid_preferred.sum().item()),
                    "selected_mean_obs": float(valid_n_obs.float().mean().item()),
                    "selected_mean_score": float(
                        selected_track_quality[valid_tracks].float().mean().item()
                    ),
                    "selected_obs_distribution": ",".join(obs_distribution),
                }
            )
        else:
            track_stats.update(
                {
                    "selected_tracks": 0,
                    "selected_anchor_tracks": 0,
                    "selected_non_anchor_tracks": 0,
                    "selected_preferred_tracks": 0,
                    "selected_mean_obs": 0.0,
                    "selected_mean_score": 0.0,
                    "selected_obs_distribution": "2:0,3:0,4:0,5+:0",
                }
            )
        if valid_tracks.sum() < self.min_num_inliers:
            return None

        valid_obs_mask = (selected_uvs[valid_tracks] >= 0).all(dim=-1)
        per_camera_observations = valid_obs_mask.sum(dim=0)
        min_camera_observations = max(20, min(self.min_num_inliers, 50))

        front_seed = torch.zeros(n_cams, dtype=torch.bool, device="cuda")
        tail_seed = torch.zeros_like(front_seed)
        for fixed_offset, keyframe in enumerate(fixed_keyframes):
            camera_id = len(movable_keyframes) + fixed_offset
            if keyframe.info.get("is_submap_seed_anchor", False):
                front_seed[camera_id] = True
            if keyframe.info.get("is_submap_tail_anchor", False):
                tail_seed[camera_id] = True
        n_movable = len(movable_keyframes)
        movable_obs_mask = valid_obs_mask[:, :n_movable]
        camera_centres = -torch.bmm(
            Rts_init[:, :3, :3].transpose(1, 2),
            Rts_init[:, :3, 3, None],
        ).squeeze(-1)
        all_front_ids = torch.where(front_seed)[0]
        if len(all_front_ids) >= 2:
            front_pair_baselines = torch.cdist(
                camera_centres[all_front_ids], camera_centres[all_front_ids]
            )
            min_front_baseline = torch.clamp(
                front_pair_baselines.max() * 1e-4, min=1e-6
            )
        else:
            min_front_baseline = torch.tensor(
                1e-6, dtype=Rts_init.dtype, device=Rts_init.device
            )

        reachable_front_anchor_count = torch.zeros(
            n_movable, dtype=torch.int64, device=Rts_init.device
        )
        reachable_tail_anchor_count = torch.zeros_like(
            reachable_front_anchor_count
        )
        reachable_front_baseline = torch.zeros(
            n_movable, dtype=Rts_init.dtype, device=Rts_init.device
        )
        min_boundary_anchor_tracks = max(4, min(self.min_num_inliers, 20))

        # Connected components must be formed only through variable cameras
        # and variable points.  A fixed camera is a boundary leaf: two tracks
        # that merely observe the same fixed camera do not share a variable and
        # must not be allowed to propagate connectivity through that camera.
        unassigned_movable = torch.ones(
            n_movable, dtype=torch.bool, device=Rts_init.device
        )
        while bool(unassigned_movable.any().item()):
            start = int(torch.where(unassigned_movable)[0][0].item())
            component_movable = torch.zeros_like(unassigned_movable)
            component_movable[start] = True
            for _ in range(n_movable):
                component_tracks = movable_obs_mask[:, component_movable].any(
                    dim=-1
                )
                expanded_movable = movable_obs_mask[component_tracks].any(dim=0)
                updated_movable = component_movable | expanded_movable
                if torch.equal(updated_movable, component_movable):
                    break
                component_movable = updated_movable
            component_tracks = movable_obs_mask[:, component_movable].any(dim=-1)
            component_camera_observations = valid_obs_mask[
                component_tracks
            ].sum(dim=0)
            strong_front_ids = torch.where(
                front_seed
                & (component_camera_observations >= min_boundary_anchor_tracks)
            )[0]
            strong_tail_ids = torch.where(
                tail_seed
                & (component_camera_observations >= min_boundary_anchor_tracks)
            )[0]

            component_front_baseline = torch.tensor(
                0.0, dtype=Rts_init.dtype, device=Rts_init.device
            )
            if len(strong_front_ids) >= 2:
                component_front_baseline = torch.cdist(
                    camera_centres[strong_front_ids],
                    camera_centres[strong_front_ids],
                ).max()
            reachable_front_anchor_count[component_movable] = len(strong_front_ids)
            reachable_tail_anchor_count[component_movable] = len(strong_tail_ids)
            reachable_front_baseline[component_movable] = component_front_baseline
            unassigned_movable[component_movable] = False

        front_reachable = reachable_front_anchor_count > 0
        tail_reachable = reachable_tail_anchor_count > 0
        movable_scale_constrained = (
            (reachable_front_anchor_count >= 2)
            & (reachable_front_baseline > min_front_baseline)
        )
        movable_well_observed = (
            per_camera_observations[:n_movable]
            >= min_camera_observations
        )
        movable_boundary_connected = (
            front_reachable[:n_movable] & movable_scale_constrained
        )
        if require_tail_connectivity:
            movable_boundary_connected = (
                movable_boundary_connected
                & tail_reachable[:n_movable]
            )
        graph_stats = {
            "min_movable_observations": int(
                per_camera_observations[:n_movable].min().item()
            ),
            "underconstrained_movable_cameras": int(
                (~movable_well_observed).sum().item()
            ),
            "scale_underconstrained_movable_cameras": int(
                (~movable_scale_constrained).sum().item()
            ),
            "min_reachable_front_anchors": int(
                reachable_front_anchor_count.min().item()
            ),
            "min_reachable_front_baseline": float(
                reachable_front_baseline.min().item()
            ),
            "min_reachable_tail_anchors": int(
                reachable_tail_anchor_count.min().item()
            ),
            "min_boundary_anchor_tracks": min_boundary_anchor_tracks,
            "required_front_baseline": float(min_front_baseline.item()),
            "boundary_disconnected_movable_cameras": int(
                (~movable_boundary_connected).sum().item()
            ),
            "explicit_tail_match_pairs": n_new_anchor_pairs,
        }
        if not movable_well_observed.all() or not movable_boundary_connected.all():
            weak_ids = torch.where(
                ~movable_well_observed | ~movable_boundary_connected
            )[0].tolist()
            weak_details = [
                (
                    movable_keyframes[i].info.get("name", movable_keyframes[i].index),
                    int(per_camera_observations[i].item()),
                    bool(front_reachable[i].item()),
                    bool(tail_reachable[i].item()),
                    int(reachable_front_anchor_count[i].item()),
                    int(reachable_tail_anchor_count[i].item()),
                    float(reachable_front_baseline[i].item()),
                )
                for i in weak_ids[:10]
            ]
            message = (
                "Anchored BA graph is underconstrained; refusing to optimize "
                "or overwrite online poses. "
                "weak cameras (name, observations, front, tail, front_anchors, "
                f"tail_anchors, front_baseline)={weak_details}"
            )
            if strict_graph:
                raise RuntimeError(message)
            print(f"[online_anchor_ba] {message}")
            return None

        with get_global_profiler().stage("final_ba.pycolmap"):
            pycolmap_result = run_pycolmap_final_ba(
                selected_uvs=selected_uvs,
                xyz_init=xyz_init,
                valid_tracks=valid_tracks,
                Rts_init=Rts_init,
                width=ordered_keyframes[0].width,
                height=ordered_keyframes[0].height,
                focal=f_init,
                centre=self.centre,
                n_movable=len(movable_keyframes),
                optimize_focal=optimize_focal,
                linear_solver=self.final_submap_ba_pycolmap_solver,
                max_num_iterations=iters,
            )
        solver_usable = bool(pycolmap_result["solution_usable"])
        solver_usable = solver_usable and all(
            (
                bool(torch.isfinite(pycolmap_result["refined_Rts"]).all().item()),
                bool(torch.isfinite(pycolmap_result["points3d"]).all().item()),
                math.isfinite(float(pycolmap_result["focal"])),
                math.isfinite(float(pycolmap_result["residual"])),
            )
        )
        if solver_usable:
            refined_Rts = pycolmap_result["refined_Rts"]
            xyz = xyz_init.clone()
            xyz[valid_tracks] = pycolmap_result["points3d"]
            f = torch.tensor(
                [pycolmap_result["focal"]],
                device="cuda",
                dtype=f_init.dtype,
            )
            final_residual = torch.tensor(
                pycolmap_result["residual"],
                device="cuda",
                dtype=f_init.dtype,
            )
        else:
            print(
                "[final_ba] Ceres reported an unusable solution; keeping "
                "all online poses unchanged. "
                f"{pycolmap_result['solver_report']}"
            )
            refined_Rts = Rts_init.clone()
            xyz = xyz_init.clone()
            f = f_init.clone()
            final_residual = torch.tensor(float("inf"), device="cuda")
        solver_report = pycolmap_result["solver_report"]
        linear_solver = pycolmap_result["linear_solver"]
        termination_type = pycolmap_result.get("termination_type")

        for keyframe, Rt in zip(movable_keyframes, refined_Rts[: len(movable_keyframes)]):
            keyframe.set_Rt(Rt)
        for keyframe in fixed_keyframes:
            keyframe.set_Rt(keyframe.info["anchor_Rt"])

        if optimize_focal and solver_usable:
            self.f = f
            self.intrinsics = torch.cat([f, self.centre], dim=0)

        points3d = xyz[valid_tracks].detach()
        points3d_rgb = None
        selected_colors = None
        if return_point_colors or return_track_groups or return_colmap_payload:
            selected_colors = self.sample_track_colors(ordered_keyframes, selected_uvs)
        if return_point_colors:
            points3d_rgb = selected_colors[
                valid_tracks
            ].detach()

        track_groups = None
        if return_track_groups:
            candidate_xyz_init, candidate_valid_tracks = self.triangulate_xyz_init(
                candidate_uvs, Rts_init, f_init
            )
            candidate_mask = selection["candidate_valid"] & candidate_valid_tracks
            selected_anchor_mask = valid_tracks & selection["selected_anchor_connected"]
            selected_non_anchor_mask = valid_tracks & ~selection["selected_anchor_connected"]
            candidate_colors = self.sample_track_colors(
                ordered_keyframes, candidate_uvs
            )[candidate_mask].detach()
            track_groups = {
                "candidate_tracks": {
                    "xyz": candidate_xyz_init[candidate_mask].detach(),
                    "rgb": candidate_colors,
                },
                "selected_anchor_connected_tracks": {
                    "xyz": xyz[selected_anchor_mask].detach(),
                    "rgb": selected_colors[selected_anchor_mask].detach(),
                },
                "selected_non_anchor_tracks": {
                    "xyz": xyz[selected_non_anchor_mask].detach(),
                    "rgb": selected_colors[selected_non_anchor_mask].detach(),
                },
            }

        colmap_payload = None
        if return_colmap_payload:
            colmap_payload = {
                "keyframes": ordered_keyframes,
                "uvs": selected_uvs[valid_tracks].detach().cpu(),
                "points3d": points3d.detach().cpu(),
                "rgb": (
                    selected_colors[valid_tracks].detach().cpu()
                    if selected_colors is not None
                    else None
                ),
            }

        return {
            "residual": final_residual.item(),
            "focal": f.item(),
            "n_movable": len(movable_keyframes),
            "n_fixed": len(fixed_keyframes),
            "backend": self.final_submap_ba_backend,
            "linear_solver": linear_solver,
            "solver_report": solver_report,
            "termination_type": termination_type,
            "solver_usable": solver_usable,
            "points3d": points3d,
            "points3d_rgb": points3d_rgb,
            "track_groups": track_groups,
            "colmap_payload": colmap_payload,
            **track_stats,
            **depth_stats,
            **graph_stats,
        }

    @torch.no_grad()
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
        debug_label=None,
        pose_init_Rt: torch.Tensor | None = None,
    ):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        
        prof = get_global_profiler()
        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        match_debug = []
        best_match_keyframe = None
        best_match_matches = None
        best_match_count = -1
        with prof.stage("incremental.matching"):
            for keyframe in keyframes:
                matches = self.matcher(
                    curr_desc_kpts,
                    keyframe.desc_kpts,
                    remove_outliers=not self.skip_incremental_magsac,
                    update_kpts_flag="inliers",
                    kID=index,
                    kID_other=keyframe.index,
                )

                mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
                matched_xyz = keyframe.desc_kpts.pts3d[matches.idx_other[mask]]
                matched_uv = matches.kpts[mask]
                matched_conf = keyframe.desc_kpts.pts_conf[matches.idx_other[mask]]
                matched_indices = matches.idx[mask]
                n_matches_3d = int(mask.sum().item())
                if len(matched_xyz) > 0:
                    xyz_ref_cam = matched_xyz @ keyframe.get_R().T + keyframe.get_t()
                    depth_mask = (
                        torch.isfinite(matched_xyz).all(dim=-1)
                        & torch.isfinite(matched_uv).all(dim=-1)
                        & torch.isfinite(matched_conf)
                        & torch.isfinite(xyz_ref_cam[:, 2])
                        & (xyz_ref_cam[:, 2] > 1e-6)
                    )
                    if self.pnp_max_depth > 0:
                        depth_mask &= xyz_ref_cam[:, 2] <= self.pnp_max_depth
                    matched_xyz = matched_xyz[depth_mask]
                    matched_uv = matched_uv[depth_mask]
                    matched_conf = matched_conf[depth_mask]
                    matched_indices = matched_indices[depth_mask]
                n_matches_used = int(len(matched_xyz))
                xyz.append(matched_xyz)
                uvs.append(matched_uv)
                confs.append(matched_conf)
                match_indices.append(matched_indices)
                if n_matches_3d > best_match_count:
                    best_match_count = n_matches_3d
                    best_match_keyframe = keyframe
                    best_match_matches = matches
                if self.debug_pnp_matching:
                    stats = getattr(self.matcher, "last_match_stats", {})
                    match_debug.append(
                        {
                            "keyframe": keyframe.index,
                            "raw": stats.get("raw_matches", len(matches.idx)),
                            "geom": stats.get("geom_inliers", len(matches.idx)),
                            "with_3d": int(mask.sum().item()),
                            "used_3d": n_matches_used,
                        }
                    )

            xyz = torch.cat(xyz, dim=0)
            uvs = torch.cat(uvs, dim=0)
            confs = torch.cat(confs, dim=0)
            match_indices = torch.cat(match_indices, dim=0)

        # The same current-frame keypoint can be returned by several reference
        # keyframes, sometimes with different 3D points.  It must vote only
        # once in PnP; otherwise a repeated feature artificially inflates the
        # inlier count and overweights a possibly inconsistent association.
        if len(match_indices) > 0:
            confidence_order = torch.argsort(
                confs, descending=True, stable=True
            )
            id_order = torch.argsort(
                match_indices[confidence_order], stable=True
            )
            unique_order = confidence_order[id_order]
            sorted_match_ids = match_indices[unique_order]
            keep_unique = torch.ones(
                len(unique_order), dtype=torch.bool, device=xyz.device
            )
            keep_unique[1:] = sorted_match_ids[1:] != sorted_match_ids[:-1]
            unique_order = unique_order[keep_unique]
            xyz = xyz[unique_order]
            uvs = uvs[unique_order]
            confs = confs[unique_order]
            match_indices = match_indices[unique_order]

        n_3d_matches = len(xyz)
        if n_3d_matches < 4:
            if self.pose_debugger is not None:
                self.pose_debugger.update_incremental(
                    debug_label if debug_label is not None else index,
                    curr_img,
                    best_match_keyframe,
                    best_match_matches,
                    xyz,
                    None,
                    None,
                    None,
                    self.f,
                )
            self._report_incremental_pnp_failure(
                index=index,
                n_3d_matches=n_3d_matches,
                n_pnp_input=n_3d_matches,
                n_pnp_inliers=0,
                n_miniba_inliers=0,
                match_debug=match_debug,
                debug_label=debug_label,
            )
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
                curr_desc_kpts.matches.pop(keyframe.index, None)
            return None

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            sampling_weights = confs.clamp_min(0)
            if sampling_weights.sum() > 0:
                selected_indices = torch.multinomial(
                    sampling_weights,
                    self.num_pts_pnpransac,
                    replacement=False,
                )
            else:
                selected_indices = torch.randperm(
                    len(xyz), device=xyz.device
                )[: self.num_pts_pnpransac]
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]
        n_pnp_input = len(xyz)
        pnp_input_xyz = xyz.detach().clone()

        # Estimate an initial camera pose and inliers using PnP RANSAC
        # P4P hypotheses are refined by a short MiniBA and therefore depend on
        # their starting pose.  Match-ranked references are not temporal-order
        # sorted (the first one can be many frames old), so initialize from the
        # latest registered camera supplied by the caller instead.
        if pose_init_Rt is None:
            pose_init_keyframe = max(keyframes, key=lambda keyframe: keyframe.index)
            Rs6D_init = pose_init_keyframe.rW2C
            ts_init = pose_init_keyframe.tW2C
        else:
            Rs6D_init = pose_init_Rt[:3, :2]
            ts_init = pose_init_Rt[:3, 3]
        with prof.stage("incremental.pnp_ransac"):
            Rt, inliers = self.PnPRANSAC(
                uvs,
                xyz,
                self.f,
                self.centre,
                Rs6D_init,
                ts_init,
                confs,
            )
        n_pnp_inliers = int(inliers.sum().item())
        pnp_Rt = Rt.detach().clone()

        if not is_test and n_pnp_inliers < self.min_num_inliers:
            if self.pose_debugger is not None:
                self.pose_debugger.update_incremental(
                    debug_label if debug_label is not None else index,
                    curr_img,
                    best_match_keyframe,
                    best_match_matches,
                    pnp_input_xyz,
                    pnp_input_xyz[inliers],
                    pnp_Rt,
                    None,
                    self.f,
                )
            self._report_incremental_pnp_failure(
                index=index,
                n_3d_matches=n_3d_matches,
                n_pnp_input=n_pnp_input,
                n_pnp_inliers=n_pnp_inliers,
                n_miniba_inliers=0,
                match_debug=match_debug,
                debug_label=debug_label,
            )
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
                curr_desc_kpts.matches.pop(keyframe.index, None)
            return None

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]
        pnp_inlier_xyz = xyz.detach().clone()

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the initialization
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        with prof.stage("incremental.miniba"):
            Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(
                Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1)
            )
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]
        if self.pose_debugger is not None:
            self.pose_debugger.update_incremental(
                debug_label if debug_label is not None else index,
                curr_img,
                best_match_keyframe,
                best_match_matches,
                pnp_input_xyz,
                pnp_inlier_xyz,
                pnp_Rt,
                Rt,
                self.f,
            )

        # Check if we have sufficiently many inliers
        n_miniba_inliers = int(
            (mask.view(-1, 2) > 0).all(dim=-1).sum().item()
        )
        if is_test or n_miniba_inliers >= self.min_num_inliers:
            # Return the pose of the current frame
            return Rt
        else:
            self._report_incremental_pnp_failure(
                index=index,
                n_3d_matches=n_3d_matches,
                n_pnp_input=n_pnp_input,
                n_pnp_inliers=n_pnp_inliers,
                n_miniba_inliers=n_miniba_inliers,
                match_debug=match_debug,
                debug_label=debug_label,
            )
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
                curr_desc_kpts.matches.pop(keyframe.index, None)
            return None

    def _report_incremental_pnp_failure(
        self,
        index: int,
        n_3d_matches: int,
        n_pnp_input: int,
        n_pnp_inliers: int,
        n_miniba_inliers: int,
        match_debug: list[dict],
        debug_label=None,
    ):
        frame_label = index if debug_label is None else debug_label
        print(
            "Too few inliers for pose initialization "
            f"(frame={frame_label}, local_index={index}, 3d_matches={n_3d_matches}, "
            f"pnp_input={n_pnp_input}, pnp_inliers={n_pnp_inliers}, "
            f"miniba_inliers={n_miniba_inliers}/{self.min_num_inliers}, "
            f"focal={float(self.f.item()):.1f})"
        )
        if not self.debug_pnp_matching:
            return
        rows = []
        for row in match_debug:
            text = (
                f"kf={row['keyframe']}:raw={row['raw']},"
                f"geom={row['geom']},3d={row['with_3d']}"
            )
            if "used_3d" in row and row["used_3d"] != row["with_3d"]:
                text += f",used={row['used_3d']}"
            rows.append(text)
        print("PnP match details: " + "; ".join(rows))
