from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.read_write_model import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
)


Rgb = tuple[int, int, int]


@dataclass(frozen=True)
class CameraRule:
    pattern: str
    scale: float | None
    color: Rgb | None


@dataclass
class CameraPose:
    image_id: int
    name: str
    camera_id: int
    width: int
    height: int
    focal_y: float
    c2w: np.ndarray


@dataclass
class ColmapModel:
    label: str
    path: Path
    cameras: dict
    poses: list[CameraPose]
    points: np.ndarray
    colors: np.ndarray


@dataclass
class ModelControls:
    label: str
    visible_cameras: object
    visible_trajectory: object
    visible_points: object
    camera_scale: object
    camera_color: object
    camera_line_width: object
    camera_rules: object
    point_size: object
    point_opacity: object
    point_use_rgb: object
    point_color: object
    apply_rules: object


class ColmapCompareViewer:
    def __init__(self, args: argparse.Namespace):
        try:
            import viser
            from viser import transforms as tf
        except ImportError as exc:
            raise ImportError(
                "This script requires `viser`. Install it in the active environment, "
                "for example: pip install viser"
            ) from exc

        self.args = args
        self.tf = tf
        self.server = viser.ViserServer(host=args.host, port=args.port, label="COLMAP Compare")
        self.server.scene.set_up_direction(args.up)
        self.server.gui.configure_theme(control_layout="collapsible")

        self.models: dict[str, ColmapModel] = {}
        self.controls: dict[str, ModelControls] = {}
        self.scene_names: dict[str, set[str]] = {}
        self.point_render = args.point_render
        if self.point_render == "gaussian" and not hasattr(self.server.scene, "add_gaussian_splats"):
            print("[viewer] add_gaussian_splats is unavailable; falling back to point cloud rendering.")
            self.point_render = "point"

        if args.grid:
            self.server.scene.add_grid(
                "/world/grid",
                width=args.grid_width,
                height=args.grid_width,
                cell_size=args.grid_cell_size,
                cell_thickness=0.25,
                section_size=args.grid_section_size,
                section_thickness=0.6,
            )

        self.models["est"] = load_colmap_model(
            args.est,
            label=args.est_label,
            sort_by=args.sort_by,
            scale=args.est_scale,
            max_points=args.max_points,
            seed=args.seed,
        )
        self.models["gt"] = load_colmap_model(
            args.gt,
            label=args.gt_label,
            sort_by=args.sort_by,
            scale=args.gt_scale,
            max_points=args.max_points,
            seed=args.seed,
        )

        if args.filter_gt_by_est:
            filter_poses_by_other_model(
                self.models["gt"],
                other=self.models["est"],
                match_by=args.filter_match_by,
                case_sensitive=args.filter_case_sensitive,
            )

        if args.align_keyword:
            scale, translation = align_model_scale_translation_from_named_poses(
                self.models,
                keyword=args.align_keyword,
                target=args.align_target,
                match_by=args.align_match_by,
                min_images=args.align_min_images,
                max_pairs=args.align_max_pairs,
                seed=args.seed,
                case_sensitive=args.align_case_sensitive,
            )
            print(
                f"[align] Applied scale {scale:.8g} and translation "
                f"[{translation[0]:.8g}, {translation[1]:.8g}, {translation[2]:.8g}] "
                f"to {args.align_target} "
                f"using keyword '{args.align_keyword}'."
            )

        if args.center:
            self._center_models()

        self._build_gui()
        self.redraw_all()

        print(
            f"[viewer] Listening at http://{args.host}:{args.port}\n"
            f"[viewer] Estimated: {args.est} ({len(self.models['est'].poses)} cameras, "
            f"{len(self.models['est'].points)} points)\n"
            f"[viewer] GT:        {args.gt} ({len(self.models['gt'].poses)} cameras, "
            f"{len(self.models['gt'].points)} points)"
        )

    def _center_models(self) -> None:
        centers = []
        for model in self.models.values():
            if model.points.size:
                centers.append(np.mean(model.points, axis=0))
            centers.extend([pose.c2w[:3, 3] for pose in model.poses])
        if not centers:
            return
        offset = np.mean(np.asarray(centers), axis=0).astype(np.float32)
        for model in self.models.values():
            if model.points.size:
                model.points = model.points - offset[None, :]
            for pose in model.poses:
                pose.c2w[:3, 3] -= offset

    def _build_gui(self) -> None:
        self.server.gui.add_markdown(
            "COLMAP SFM compare viewer\n\n"
            "- Estimated and GT cameras/points can be controlled independently.\n"
            "- Camera rules are one per line: `name_substring scale #RRGGBB`.\n"
            "- Rules override the base frustum scale/color when the image name contains the substring.\n"
            "- Point opacity uses gaussian splats when available; point-cloud fallback ignores true opacity."
        )

        defaults = {
            "est": {
                "camera_color": self.args.est_camera_color,
                "camera_scale": self.args.est_camera_scale,
                "point_color": self.args.est_point_color,
                "point_size": self.args.est_point_size,
                "point_opacity": self.args.est_point_opacity,
                "rules": self.args.est_camera_rule,
            },
            "gt": {
                "camera_color": self.args.gt_camera_color,
                "camera_scale": self.args.gt_camera_scale,
                "point_color": self.args.gt_point_color,
                "point_size": self.args.gt_point_size,
                "point_opacity": self.args.gt_point_opacity,
                "rules": self.args.gt_camera_rule,
            },
        }

        for key in ("est", "gt"):
            model = self.models[key]
            init = defaults[key]
            with self.server.gui.add_folder(model.label, expand_by_default=True):
                visible_cameras = self.server.gui.add_checkbox("Show cameras", True)
                visible_trajectory = self.server.gui.add_checkbox("Show trajectory", True)
                visible_points = self.server.gui.add_checkbox("Show points", key == "gt")
                camera_scale = self.server.gui.add_slider(
                    "Frustum scale",
                    min=0.0,
                    max=max(1.0, init["camera_scale"] * 10.0),
                    step=0.001,
                    initial_value=float(init["camera_scale"]),
                )
                camera_color = self.server.gui.add_text(
                    "Camera color",
                    initial_value=rgb_to_hex(parse_color(init["camera_color"])),
                )
                camera_line_width = self.server.gui.add_slider(
                    "Frustum line width",
                    min=0.1,
                    max=8.0,
                    step=0.1,
                    initial_value=2.0,
                )
                camera_rules = self.server.gui.add_text(
                    "Name rules",
                    initial_value="\n".join(init["rules"]),
                    multiline=True,
                    hint="Format: name_substring scale #RRGGBB",
                )
                apply_rules = self.server.gui.add_button("Apply camera rules")
                point_size = self.server.gui.add_slider(
                    "Point size",
                    min=0.0001,
                    max=max(0.2, init["point_size"] * 20.0),
                    step=0.0005,
                    initial_value=float(init["point_size"]),
                )
                point_opacity = self.server.gui.add_slider(
                    "Point opacity",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(init["point_opacity"]),
                )
                point_use_rgb = self.server.gui.add_checkbox("Use COLMAP point RGB", True)
                point_color = self.server.gui.add_text(
                    "Point fallback color",
                    initial_value=rgb_to_hex(parse_color(init["point_color"])),
                )

            controls = ModelControls(
                label=key,
                visible_cameras=visible_cameras,
                visible_trajectory=visible_trajectory,
                visible_points=visible_points,
                camera_scale=camera_scale,
                camera_color=camera_color,
                camera_line_width=camera_line_width,
                camera_rules=camera_rules,
                point_size=point_size,
                point_opacity=point_opacity,
                point_use_rgb=point_use_rgb,
                point_color=point_color,
                apply_rules=apply_rules,
            )
            self.controls[key] = controls

            for handle in (
                visible_cameras,
                visible_trajectory,
                camera_scale,
                camera_color,
                camera_line_width,
            ):
                handle.on_update(lambda _event, model_key=key: self.redraw_cameras(model_key))
            apply_rules.on_click(lambda _event, model_key=key: self.redraw_cameras(model_key))
            for handle in (visible_points, point_size, point_opacity, point_use_rgb, point_color):
                handle.on_update(lambda _event, model_key=key: self.redraw_points(model_key))

    def redraw_all(self) -> None:
        for key in self.models:
            self.redraw_cameras(key)
            self.redraw_points(key)
        self.server.flush()

    def redraw_cameras(self, key: str) -> None:
        model = self.models[key]
        controls = self.controls[key]
        group = f"{key}_cameras"
        self._clear_group(group)

        visible_cameras = bool(controls.visible_cameras.value)
        visible_trajectory = bool(controls.visible_trajectory.value)
        base_scale = float(controls.camera_scale.value)
        base_color = parse_color(str(controls.camera_color.value))
        line_width = float(controls.camera_line_width.value)
        rules = parse_rules(str(controls.camera_rules.value))

        names = self.scene_names.setdefault(group, set())
        centers = []
        for idx, pose in enumerate(model.poses):
            scale, color = apply_camera_rules(pose.name, base_scale, base_color, rules)
            node_name = f"/colmap_compare/{key}/cameras/cam_{idx:05d}_{safe_name(pose.name)}"
            self._add_camera_frustum(
                node_name,
                pose,
                color=color,
                scale=scale,
                line_width=line_width,
                visible=visible_cameras,
            )
            names.add(node_name)
            centers.append(pose.c2w[:3, 3].astype(np.float32))

        if len(centers) >= 2:
            points = np.asarray(centers, dtype=np.float32)
            trajectory_name = f"/colmap_compare/{key}/trajectory"
            self.server.scene.add_line_segments(
                trajectory_name,
                np.stack([points[:-1], points[1:]], axis=1),
                np.asarray(base_color, dtype=np.uint8),
                line_width=max(1.0, line_width),
                visible=visible_trajectory,
            )
            names.add(trajectory_name)
        self.server.flush()

    def redraw_points(self, key: str) -> None:
        model = self.models[key]
        controls = self.controls[key]
        group = f"{key}_points"
        self._clear_group(group)

        points = model.points
        if points.size == 0:
            return

        visible = bool(controls.visible_points.value)
        point_size = float(controls.point_size.value)
        opacity = float(controls.point_opacity.value)
        fallback_color = np.asarray(parse_color(str(controls.point_color.value)), dtype=np.uint8)
        if bool(controls.point_use_rgb.value) and model.colors.shape == points.shape:
            colors = model.colors
        else:
            colors = np.repeat(fallback_color[None, :], len(points), axis=0)

        name = f"/colmap_compare/{key}/points"
        if self.point_render == "gaussian":
            rgb_float = np.clip(colors.astype(np.float32) / 255.0, 0.0, 1.0)
            opacities = np.full((len(points), 1), np.clip(opacity, 0.0, 1.0), dtype=np.float32)
            covariances = np.repeat(
                (np.eye(3, dtype=np.float32) * max(point_size, 1e-6) ** 2)[None, :, :],
                len(points),
                axis=0,
            )
            self.server.scene.add_gaussian_splats(
                name,
                centers=points.astype(np.float32),
                rgbs=rgb_float,
                opacities=opacities,
                covariances=covariances,
                visible=visible,
            )
        else:
            display_colors = blend_for_opacity(colors, opacity)
            self.server.scene.add_point_cloud(
                name,
                points.astype(np.float32),
                display_colors,
                point_size=point_size,
                point_shape="circle",
                visible=visible,
            )
        self.scene_names.setdefault(group, set()).add(name)
        self.server.flush()

    def _add_camera_frustum(
        self,
        name: str,
        pose: CameraPose,
        color: Rgb,
        scale: float,
        line_width: float,
        visible: bool,
    ) -> None:
        rotation = self.tf.SO3.from_matrix(pose.c2w[:3, :3])
        fov = 2.0 * math.atan(pose.height / (2.0 * max(float(pose.focal_y), 1e-6)))
        self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=pose.width / max(pose.height, 1),
            scale=scale,
            line_width=line_width,
            color=color,
            wxyz=rotation.wxyz,
            position=pose.c2w[:3, 3],
            visible=visible,
        )

    def _clear_group(self, group: str) -> None:
        for name in self.scene_names.get(group, set()):
            self.server.scene.remove_by_name(name)
        self.scene_names[group] = set()

    def run(self) -> None:
        while True:
            time.sleep(0.05)


def load_colmap_model(
    path: str | os.PathLike[str],
    label: str,
    sort_by: str,
    scale: float,
    max_points: int,
    seed: int,
) -> ColmapModel:
    model_path = resolve_colmap_model_dir(Path(path))
    cameras, images, points3D = read_colmap_model_allow_missing_points(model_path)

    image_items = list(images.items())
    if sort_by == "name":
        image_items.sort(key=lambda item: item[1].name)
    else:
        image_items.sort(key=lambda item: item[0])

    poses: list[CameraPose] = []
    for image_id, image in image_items:
        camera = cameras[image.camera_id]
        c2w = world_to_camera_to_camera_to_world(image.qvec, image.tvec, scale)
        poses.append(
            CameraPose(
                image_id=int(image_id),
                name=str(image.name),
                camera_id=int(image.camera_id),
                width=int(camera.width),
                height=int(camera.height),
                focal_y=camera_focal_y(camera),
                c2w=c2w,
            )
        )

    points = np.empty((0, 3), dtype=np.float32)
    colors = np.empty((0, 3), dtype=np.uint8)
    if points3D:
        point_items = sorted(points3D.items())
        points = np.asarray([point.xyz for _, point in point_items], dtype=np.float32) * float(scale)
        colors = np.asarray([point.rgb for _, point in point_items], dtype=np.uint8)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        colors = colors[valid]
        if max_points > 0 and len(points) > max_points:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(points), size=max_points, replace=False))
            points = points[keep]
            colors = colors[keep]

    return ColmapModel(
        label=label,
        path=model_path,
        cameras=cameras,
        poses=poses,
        points=points,
        colors=colors,
    )


def resolve_colmap_model_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"COLMAP model directory not found: {path}")
    if has_colmap_cameras_and_images(path):
        return path
    sparse_zero = path / "0"
    if sparse_zero.is_dir() and has_colmap_cameras_and_images(sparse_zero):
        print(f"[colmap] Using nested sparse model directory: {sparse_zero}")
        return sparse_zero
    raise ValueError(
        f"Could not find cameras/images COLMAP files in {path}. "
        "Expected cameras.bin/images.bin or cameras.txt/images.txt."
    )


def has_colmap_cameras_and_images(path: Path) -> bool:
    return any(
        (path / f"cameras{ext}").is_file() and (path / f"images{ext}").is_file()
        for ext in (".bin", ".txt")
    )


def read_colmap_model_allow_missing_points(path: Path) -> tuple[dict, dict, dict]:
    if (path / "cameras.bin").is_file() and (path / "images.bin").is_file():
        ext = ".bin"
        cameras = read_cameras_binary(str(path / "cameras.bin"))
        images = read_images_binary(str(path / "images.bin"))
        points_path = path / "points3D.bin"
        points3D = read_points3D_binary(str(points_path)) if points_path.is_file() else {}
    elif (path / "cameras.txt").is_file() and (path / "images.txt").is_file():
        ext = ".txt"
        cameras = read_cameras_text(str(path / "cameras.txt"))
        images = read_images_text(str(path / "images.txt"))
        points_path = path / "points3D.txt"
        points3D = read_points3D_text(str(points_path)) if points_path.is_file() else {}
    else:
        raise ValueError(
            f"Could not find cameras/images COLMAP files in {path}. "
            "Expected cameras.bin/images.bin or cameras.txt/images.txt."
        )

    if not points_path.is_file():
        print(f"[colmap] {points_path.name} not found in {path}; continuing without sparse points.")
    print(
        f"[colmap] Loaded {path} ({ext}): "
        f"{len(cameras)} cameras, {len(images)} images, {len(points3D)} points."
    )
    return cameras, images, points3D


def world_to_camera_to_camera_to_world(qvec: np.ndarray, tvec: np.ndarray, scale: float) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(qvec)
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64) * float(scale)
    c2w = np.linalg.inv(w2c)
    return c2w.astype(np.float32)


def camera_focal_y(camera) -> float:
    params = np.asarray(camera.params, dtype=np.float64)
    if len(params) == 0:
        return float(max(camera.width, camera.height))
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"} and len(params) > 1:
        return float(params[1])
    return float(params[0])


def align_model_scale_translation_from_named_poses(
    models: dict[str, ColmapModel],
    keyword: str,
    target: str,
    match_by: str,
    min_images: int,
    max_pairs: int,
    seed: int,
    case_sensitive: bool,
) -> tuple[float, np.ndarray]:
    if target not in {"est", "gt"}:
        raise ValueError("--align-target must be 'est' or 'gt'.")
    reference = "gt" if target == "est" else "est"

    target_matches = collect_matching_poses(
        models[target].poses,
        keyword=keyword,
        match_by=match_by,
        case_sensitive=case_sensitive,
    )
    reference_matches = collect_matching_poses(
        models[reference].poses,
        keyword=keyword,
        match_by=match_by,
        case_sensitive=case_sensitive,
    )
    common_names = sorted(set(target_matches) & set(reference_matches))
    if len(common_names) < min_images:
        raise ValueError(
            f"Only found {len(common_names)} shared images matching keyword '{keyword}', "
            f"but --align-min-images={min_images}. Try --align-match-by full/basename/stem "
            "or use a broader keyword."
        )

    target_centers = np.asarray(
        [target_matches[name].c2w[:3, 3] for name in common_names],
        dtype=np.float64,
    )
    reference_centers = np.asarray(
        [reference_matches[name].c2w[:3, 3] for name in common_names],
        dtype=np.float64,
    )
    scale = estimate_distance_scale(
        source_centers=target_centers,
        reference_centers=reference_centers,
        max_pairs=max_pairs,
        seed=seed,
    )
    translation = estimate_translation(
        source_centers=target_centers,
        reference_centers=reference_centers,
        scale=scale,
    )
    apply_model_similarity(models[target], scale=scale, translation=translation)
    print(
        f"[align] {target} -> {reference}: matched {len(common_names)} images, "
        f"median pairwise distance ratio={scale:.8g}, "
        f"translation=[{translation[0]:.8g}, {translation[1]:.8g}, {translation[2]:.8g}]."
    )
    return scale, translation


def collect_matching_poses(
    poses: Iterable[CameraPose],
    keyword: str,
    match_by: str,
    case_sensitive: bool,
) -> dict[str, CameraPose]:
    keyword_cmp = keyword if case_sensitive else keyword.lower()
    matches: dict[str, CameraPose] = {}
    duplicates: set[str] = set()
    for pose in poses:
        name_cmp = pose.name if case_sensitive else pose.name.lower()
        if keyword_cmp not in name_cmp:
            continue
        key = pose_match_key(pose.name, match_by)
        if key in matches:
            duplicates.add(key)
            continue
        matches[key] = pose
    for key in duplicates:
        matches.pop(key, None)
    if duplicates:
        print(
            f"[align] Skipped {len(duplicates)} ambiguous duplicated match keys. "
            "Use --align-match-by full if this removes too many images."
        )
    return matches


def pose_match_key(name: str, match_by: str) -> str:
    path = Path(name)
    if match_by == "full":
        return name
    if match_by == "basename":
        return path.name
    if match_by == "stem":
        return path.stem
    raise ValueError("match_by must be one of: full, basename, stem.")


def filter_poses_by_other_model(
    target: ColmapModel,
    other: ColmapModel,
    match_by: str,
    case_sensitive: bool,
) -> None:
    """Drop poses from ``target`` whose image name does not appear in ``other``."""

    def name_key(name: str) -> str:
        key = pose_match_key(name, match_by)
        return key if case_sensitive else key.lower()

    other_keys = {name_key(pose.name) for pose in other.poses}
    if not other_keys:
        print(
            f"[filter] {other.label} has no poses; skipping filter on {target.label}."
        )
        return

    kept = [pose for pose in target.poses if name_key(pose.name) in other_keys]
    dropped = len(target.poses) - len(kept)
    target.poses = kept
    print(
        f"[filter] {target.label}: kept {len(kept)} poses present in {other.label}, "
        f"dropped {dropped} (match_by={match_by}, "
        f"case_sensitive={'yes' if case_sensitive else 'no'})."
    )


def estimate_distance_scale(
    source_centers: np.ndarray,
    reference_centers: np.ndarray,
    max_pairs: int,
    seed: int,
) -> float:
    if len(source_centers) != len(reference_centers):
        raise ValueError("source_centers and reference_centers must have the same length.")
    if len(source_centers) < 2:
        raise ValueError("At least two matched cameras are required to estimate scale.")

    num_centers = len(source_centers)
    total_pairs = num_centers * (num_centers - 1) // 2
    if max_pairs <= 0 or total_pairs <= max_pairs:
        pair_indices = np.asarray(
            [(i, j) for i in range(num_centers) for j in range(i + 1, num_centers)],
            dtype=np.int64,
        )
    else:
        rng = np.random.default_rng(seed)
        first = rng.integers(0, num_centers, size=max_pairs * 2)
        second = rng.integers(0, num_centers, size=max_pairs * 2)
        valid = first != second
        first = first[valid]
        second = second[valid]
        lo = np.minimum(first, second)
        hi = np.maximum(first, second)
        pair_indices = np.unique(np.stack([lo, hi], axis=1), axis=0)[:max_pairs]

    src_dist = np.linalg.norm(
        source_centers[pair_indices[:, 0]] - source_centers[pair_indices[:, 1]],
        axis=1,
    )
    ref_dist = np.linalg.norm(
        reference_centers[pair_indices[:, 0]] - reference_centers[pair_indices[:, 1]],
        axis=1,
    )
    valid = np.isfinite(src_dist) & np.isfinite(ref_dist) & (src_dist > 1e-8) & (ref_dist > 1e-8)
    ratios = ref_dist[valid] / src_dist[valid]
    if ratios.size == 0:
        raise ValueError("Matched cameras are degenerate; no valid pairwise distances for scale.")
    return float(np.median(ratios))


def estimate_translation(
    source_centers: np.ndarray,
    reference_centers: np.ndarray,
    scale: float,
) -> np.ndarray:
    residuals = reference_centers - source_centers * float(scale)
    return np.median(residuals, axis=0).astype(np.float32)


def apply_model_similarity(model: ColmapModel, scale: float, translation: np.ndarray) -> None:
    scale = float(scale)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    if model.points.size:
        model.points = model.points * scale + translation
    for pose in model.poses:
        pose.c2w[:3, 3] = pose.c2w[:3, 3] * scale + translation.reshape(3)


def parse_color(value: str | Rgb) -> Rgb:
    if isinstance(value, tuple):
        return tuple(int(np.clip(v, 0, 255)) for v in value)  # type: ignore[return-value]

    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 6:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)

    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Invalid color '{value}'. Use #RRGGBB or 'R G B'.")
    return tuple(int(np.clip(float(part), 0, 255)) for part in parts)  # type: ignore[return-value]


def rgb_to_hex(color: Rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*parse_color(color))


def parse_rules(lines: str | Iterable[str]) -> list[CameraRule]:
    if isinstance(lines, str):
        raw_lines = lines.splitlines()
    else:
        raw_lines = list(lines)

    rules = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(
                f"Invalid camera rule '{raw_line}'. Expected: name_substring scale [#RRGGBB]"
            )
        pattern = parts[0]
        scale = None if parts[1] in {"-", "none", "None"} else float(parts[1])
        color = parse_color(parts[2]) if len(parts) >= 3 and parts[2] not in {"-", "none", "None"} else None
        rules.append(CameraRule(pattern=pattern, scale=scale, color=color))
    return rules


def apply_camera_rules(
    image_name: str,
    base_scale: float,
    base_color: Rgb,
    rules: Iterable[CameraRule],
) -> tuple[float, Rgb]:
    scale = base_scale
    color = base_color
    for rule in rules:
        if rule.pattern in image_name:
            if rule.scale is not None:
                scale = rule.scale
            if rule.color is not None:
                color = rule.color
    return scale, color


def blend_for_opacity(colors: np.ndarray, opacity: float) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    background = np.full_like(colors, 20)
    blended = colors.astype(np.float32) * opacity + background.astype(np.float32) * (1.0 - opacity)
    return np.clip(blended, 0, 255).astype(np.uint8)


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("#", "_")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize estimated and GT COLMAP SFM models together in viser."
    )
    parser.add_argument("--est", required=True, help="Estimated COLMAP model directory.")
    parser.add_argument("--gt", required=True, help="GT COLMAP model directory.")
    parser.add_argument("--host", default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--port", type=int, default=8091, help="Viser server port.")
    parser.add_argument("--up", default="+z", choices=["+x", "+y", "+z", "-x", "-y", "-z"])
    parser.add_argument("--sort-by", choices=["name", "id"], default="name")
    parser.add_argument("--max-points", type=int, default=100000, help="Maximum points per model. <=0 keeps all.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point subsampling.")
    parser.add_argument("--center", action="store_true", help="Center the combined scene for easier viewing.")
    parser.add_argument("--grid", action="store_true", help="Show a world grid.")
    parser.add_argument("--grid-width", type=float, default=10.0)
    parser.add_argument("--grid-cell-size", type=float, default=0.1)
    parser.add_argument("--grid-section-size", type=float, default=1.0)
    parser.add_argument(
        "--align-keyword",
        default="",
        help=(
            "If set, estimate scale and translation from shared image names containing "
            "this keyword and apply them before visualization."
        ),
    )
    parser.add_argument(
        "--align-target",
        choices=["est", "gt"],
        default="est",
        help="Model to rescale during keyword-based alignment. Defaults to scaling est to gt.",
    )
    parser.add_argument(
        "--align-match-by",
        choices=["basename", "stem", "full"],
        default="basename",
        help="How to match shared image names after keyword filtering.",
    )
    parser.add_argument(
        "--align-min-images",
        type=int,
        default=2,
        help="Minimum number of shared keyword-matched images required for scale alignment.",
    )
    parser.add_argument(
        "--align-max-pairs",
        type=int,
        default=200000,
        help="Maximum matched camera pairs used for robust scale estimation. <=0 uses all pairs.",
    )
    parser.add_argument(
        "--align-case-sensitive",
        action="store_true",
        help="Make --align-keyword matching case-sensitive.",
    )
    parser.add_argument(
        "--filter-gt-by-est",
        action="store_true",
        help="Drop GT poses whose image name does not appear in the estimated model.",
    )
    parser.add_argument(
        "--filter-match-by",
        choices=["basename", "stem", "full"],
        default="basename",
        help="How to compare image names when filtering GT by est.",
    )
    parser.add_argument(
        "--filter-case-sensitive",
        action="store_true",
        help="Make --filter-gt-by-est matching case-sensitive.",
    )
    parser.add_argument(
        "--point-render",
        choices=["gaussian", "point"],
        default="gaussian",
        help="Use gaussian splats for true opacity, or regular point clouds for lighter rendering.",
    )

    parser.add_argument("--est-label", default="Estimated")
    parser.add_argument("--gt-label", default="GT")
    parser.add_argument("--est-scale", type=float, default=1.0)
    parser.add_argument("--gt-scale", type=float, default=1.0)
    parser.add_argument("--est-camera-scale", type=float, default=0.08)
    parser.add_argument("--gt-camera-scale", type=float, default=0.08)
    parser.add_argument("--est-camera-color", default="#4C78A8")
    parser.add_argument("--gt-camera-color", default="#E45756")
    parser.add_argument("--est-point-size", type=float, default=0.006)
    parser.add_argument("--gt-point-size", type=float, default=0.006)
    parser.add_argument("--est-point-opacity", type=float, default=0.45)
    parser.add_argument("--gt-point-opacity", type=float, default=0.45)
    parser.add_argument("--est-point-color", default="#4C78A8")
    parser.add_argument("--gt-point-color", default="#E45756")
    parser.add_argument(
        "--est-camera-rule",
        action="append",
        default=[],
        help="Estimated camera override rule: 'name_substring scale #RRGGBB'. Can be repeated.",
    )
    parser.add_argument(
        "--gt-camera-rule",
        action="append",
        default=[],
        help="GT camera override rule: 'name_substring scale #RRGGBB'. Can be repeated.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    viewer = ColmapCompareViewer(args)
    viewer.run()


if __name__ == "__main__":
    main()
