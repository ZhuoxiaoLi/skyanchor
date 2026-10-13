from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=1)
def _get_pycolmap():
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "The 'pycolmap' final BA backend requires pycolmap in the active Python environment."
        ) from exc
    return pycolmap


@lru_cache(maxsize=1)
def _get_pyceres():
    try:
        import pyceres
    except ImportError as exc:
        raise RuntimeError(
            "The 'pycolmap' final BA backend requires pyceres in the active Python environment."
        ) from exc
    return pyceres


def _torch_scalar(value) -> float:
    return float(torch.as_tensor(value).reshape(-1)[0].item())


def _build_reconstruction(
    selected_uvs: torch.Tensor,
    xyz_init: torch.Tensor,
    valid_tracks: torch.Tensor,
    Rts_init: torch.Tensor,
    width: int,
    height: int,
    focal: float,
    centre: torch.Tensor,
):
    pycolmap = _get_pycolmap()

    selected_uvs_np = selected_uvs.detach().cpu().numpy()
    xyz_init_np = xyz_init.detach().cpu().numpy()
    valid_tracks_np = valid_tracks.detach().cpu().numpy().astype(bool)
    Rts_init_np = Rts_init.detach().cpu().numpy()
    cx, cy = centre.detach().cpu().numpy().astype(np.float64)

    valid_uvs = selected_uvs_np[valid_tracks_np]
    valid_xyz = xyz_init_np[valid_tracks_np]
    n_valid, n_cams = valid_uvs.shape[:2]

    rec = pycolmap.Reconstruction()
    camera = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=np.array([focal, cx, cy], dtype=np.float64),
        camera_id=1,
    )
    rec.add_camera(camera)

    point2d_indices = -np.ones((n_valid, n_cams), dtype=np.int32)
    image_points2d: list[list[object]] = [[] for _ in range(n_cams)]
    image_ids = []

    for cam_idx in range(n_cams):
        image_id = cam_idx + 1
        image_ids.append(image_id)
        for track_idx in range(n_valid):
            uv = valid_uvs[track_idx, cam_idx]
            if np.any(uv < 0):
                continue
            point2d_indices[track_idx, cam_idx] = len(image_points2d[cam_idx])
            image_points2d[cam_idx].append(pycolmap.Point2D(xy=uv.astype(np.float64)))

        pose = pycolmap.Rigid3d(Rts_init_np[cam_idx, :3, :4].astype(np.float64))
        image = pycolmap.Image(
            name=f"submap_{image_id:04d}.png",
            points2D=image_points2d[cam_idx],
            cam_from_world=pose,
            camera_id=1,
            image_id=image_id,
        )
        rec.add_image(image)
        rec.register_image(image_id)

    point3d_ids = []
    for track_idx, xyz in enumerate(valid_xyz):
        track = pycolmap.Track()
        for cam_idx, image_id in enumerate(image_ids):
            point2d_idx = point2d_indices[track_idx, cam_idx]
            if point2d_idx >= 0:
                track.add_element(int(image_id), int(point2d_idx))
        point3d_ids.append(
            rec.add_point3D(
                xyz.astype(np.float64),
                track,
                color=np.array([0, 0, 0], dtype=np.uint8),
            )
        )

    return rec, image_ids, point3d_ids, valid_uvs, point2d_indices


def _solve_bundle_adjustment(
    rec,
    image_ids: list[int],
    point3d_ids: list[int],
    n_movable: int,
    optimize_focal: bool,
    linear_solver: str,
    max_num_iterations: int,
):
    pycolmap = _get_pycolmap()
    pyceres = _get_pyceres()

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in image_ids:
        config.add_image(int(image_id))
    for point3d_id in point3d_ids:
        config.add_variable_point(int(point3d_id))
    if not optimize_focal:
        config.set_constant_cam_intrinsics(1)

    options = pycolmap.BundleAdjustmentOptions(
        loss_function_type=pycolmap.LossFunctionType.SOFT_L1,
        loss_function_scale=1.5,
    )
    bundler = pycolmap.create_default_bundle_adjuster(options, config, rec)
    problem = bundler.problem
    quat_manifold = pyceres.EigenQuaternionManifold()

    for cam_idx, image_id in enumerate(image_ids):
        pose = rec.image(int(image_id)).cam_from_world
        if cam_idx >= n_movable:
            if problem.has_parameter_block(pose.rotation.quat):
                problem.set_parameter_block_constant(pose.rotation.quat)
            if problem.has_parameter_block(pose.translation):
                problem.set_parameter_block_constant(pose.translation)
        else:
            if problem.has_parameter_block(pose.rotation.quat):
                problem.set_manifold(pose.rotation.quat, quat_manifold)

    if optimize_focal and problem.has_parameter_block(rec.camera(1).params):
        problem.set_manifold(rec.camera(1).params, pyceres.SubsetManifold(3, [1, 2]))

    solver_options = pyceres.SolverOptions()
    solver_map = {
        "dense-schur": pyceres.LinearSolverType.DENSE_SCHUR,
        "iterative-schur": pyceres.LinearSolverType.ITERATIVE_SCHUR,
        "sparse-schur": pyceres.LinearSolverType.SPARSE_SCHUR,
        "dense-qr": pyceres.LinearSolverType.DENSE_QR,
        "sparse-normal-cholesky": pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY,
    }
    solver_options.linear_solver_type = solver_map[linear_solver]
    solver_options.max_num_iterations = max(1, int(max_num_iterations))
    if linear_solver == "iterative-schur":
        solver_options.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
    solver_options.minimizer_progress_to_stdout = False
    solver_options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    return summary


def _compute_mean_reprojection_error(
    rec,
    image_ids: list[int],
    point3d_ids: list[int],
    valid_uvs: np.ndarray,
    point2d_indices: np.ndarray,
) -> float:
    camera = rec.camera(1)
    focal, cx, cy = camera.params.astype(np.float64)

    err_sum = 0.0
    n_obs = 0
    for track_idx, point3d_id in enumerate(point3d_ids):
        xyz = rec.point3D(int(point3d_id)).xyz.astype(np.float64)
        for cam_idx, image_id in enumerate(image_ids):
            if point2d_indices[track_idx, cam_idx] < 0:
                continue
            pose = rec.image(int(image_id)).cam_from_world
            xyz_cam = pose.rotation.matrix() @ xyz + pose.translation
            if xyz_cam[2] <= 1e-12:
                continue
            uv_proj = np.array(
                [
                    focal * xyz_cam[0] / xyz_cam[2] + cx,
                    focal * xyz_cam[1] / xyz_cam[2] + cy,
                ],
                dtype=np.float64,
            )
            err_sum += float(np.linalg.norm(uv_proj - valid_uvs[track_idx, cam_idx]))
            n_obs += 1
    return err_sum / max(n_obs, 1)


def run_final_ba(
    selected_uvs: torch.Tensor,
    xyz_init: torch.Tensor,
    valid_tracks: torch.Tensor,
    Rts_init: torch.Tensor,
    width: int,
    height: int,
    focal: torch.Tensor | float,
    centre: torch.Tensor,
    n_movable: int,
    optimize_focal: bool = False,
    linear_solver: str = "dense-schur",
    max_num_iterations: int = 100,
):
    rec, image_ids, point3d_ids, valid_uvs, point2d_indices = _build_reconstruction(
        selected_uvs=selected_uvs,
        xyz_init=xyz_init,
        valid_tracks=valid_tracks,
        Rts_init=Rts_init,
        width=width,
        height=height,
        focal=_torch_scalar(focal),
        centre=centre,
    )
    summary = _solve_bundle_adjustment(
        rec=rec,
        image_ids=image_ids,
        point3d_ids=point3d_ids,
        n_movable=n_movable,
        optimize_focal=optimize_focal,
        linear_solver=linear_solver,
        max_num_iterations=max_num_iterations,
    )

    refined_Rts = []
    for image_id in image_ids:
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :4] = rec.image(int(image_id)).cam_from_world.matrix().astype(np.float32)
        refined_Rts.append(torch.from_numpy(Rt))

    optimized_points = []
    for point3d_id in point3d_ids:
        optimized_points.append(torch.from_numpy(rec.point3D(int(point3d_id)).xyz.astype(np.float32)))

    residual = _compute_mean_reprojection_error(
        rec=rec,
        image_ids=image_ids,
        point3d_ids=point3d_ids,
        valid_uvs=valid_uvs,
        point2d_indices=point2d_indices,
    )

    return {
        "refined_Rts": torch.stack(refined_Rts, dim=0).to(device=Rts_init.device, dtype=Rts_init.dtype),
        "points3d": torch.stack(optimized_points, dim=0).to(device=xyz_init.device, dtype=xyz_init.dtype),
        "focal": float(rec.camera(1).params[0]),
        "residual": float(residual),
        "linear_solver": linear_solver,
        "max_num_iterations": int(max_num_iterations),
        "solver_report": summary.BriefReport(),
        "solution_usable": bool(summary.IsSolutionUsable()),
        "termination_type": str(summary.termination_type),
    }
