#!/usr/bin/env python3
"""Evaluate anchored pose submaps against a COLMAP ground-truth model.

The evaluator is read-only unless ``--json`` or ``--csv`` is explicitly used.
It joins poses by image name, converts COLMAP world-to-camera ``qvec/tvec`` to
camera centres, and derives submap boundaries from the ordered anchor file.

Example::

    python scripts/evaluate_pose_submaps.py \
        --images-dir data/staff/images \
        --gt data/staff/gt/sparse/0 \
        --est output/staff/final_ba_colmap \
        --anchors data/staff/sub_anchors.txt \
        --anchor-front 6 --anchor-back 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NATURAL_PARTS = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    centre: np.ndarray
    observations: int
    valid_observations: int


@dataclass(frozen=True)
class AnchorPose:
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    centre: np.ndarray


@dataclass(frozen=True)
class Segment:
    index: int
    front: tuple[str, ...]
    back: tuple[str, ...]
    image_names: tuple[str, ...]
    used_boundary_fallback: bool


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(qvec))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid quaternion: {qvec.tolist()}")
    w, x, y, z = qvec / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_image_pose(
    image_id: int,
    name: str,
    qvec: np.ndarray,
    tvec: np.ndarray,
    observations: int,
    valid_observations: int,
) -> ImagePose:
    rotation = quaternion_to_rotation(qvec)
    return ImagePose(
        image_id=image_id,
        name=name,
        qvec=qvec,
        tvec=tvec,
        rotation=rotation,
        centre=-rotation.T @ tvec,
        observations=observations,
        valid_observations=valid_observations,
    )


def resolve_model_paths(model: Path, require_points: bool = False) -> tuple[Path, Path | None]:
    model = model.expanduser().resolve()
    if model.is_dir():
        images_path = model / "images.txt"
        points_path = model / "points3D.txt"
    elif model.name == "images.txt":
        images_path = model
        points_path = model.with_name("points3D.txt")
    else:
        raise ValueError(f"Expected a COLMAP text-model directory or images.txt, got: {model}")
    if not images_path.is_file():
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_path}")
    if not points_path.is_file():
        if require_points:
            raise FileNotFoundError(f"COLMAP points3D.txt not found: {points_path}")
        points_path = None
    return images_path, points_path


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def read_colmap_images(path: Path) -> tuple[dict[str, ImagePose], dict[int, str]]:
    """Read COLMAP text images, including per-image valid observation counts."""
    rows: dict[str, ImagePose] = {}
    id_to_name: dict[int, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        fields = lines[index].split()
        if len(fields) >= 10 and is_image_name(fields[9]):
            try:
                image_id = int(fields[0])
                qvec = np.asarray(fields[1:5], dtype=float)
                tvec = np.asarray(fields[5:8], dtype=float)
                name = fields[9]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed image record at {path}:{index + 1}") from exc
            point_fields = lines[index + 1].split() if index + 1 < len(lines) else []
            observations = len(point_fields) // 3
            try:
                valid_observations = sum(
                    int(point_fields[offset]) >= 0 for offset in range(2, len(point_fields), 3)
                )
            except ValueError as exc:
                raise ValueError(f"Malformed POINTS2D record at {path}:{index + 2}") from exc
            if name in rows:
                raise ValueError(f"Duplicate COLMAP image name {name!r} in {path}")
            if image_id in id_to_name:
                raise ValueError(f"Duplicate COLMAP image id {image_id} in {path}")
            rows[name] = make_image_pose(
                image_id,
                name,
                qvec,
                tvec,
                observations,
                valid_observations,
            )
            id_to_name[image_id] = name
            index += 2
        else:
            index += 1
    if not rows:
        raise ValueError(f"No image poses found in {path}")
    return rows, id_to_name


def read_anchors(path: Path) -> tuple[dict[str, AnchorPose], list[str]]:
    """Read ``image_name tx ty tz qw qx qy qz`` anchor records."""
    path = path.expanduser().resolve()
    poses: dict[str, AnchorPose] = {}
    order: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(
                f"Expected 8 anchor fields at {path}:{line_number}, got {len(fields)}"
            )
        name = fields[0]
        if name in poses:
            raise ValueError(f"Duplicate anchor image name {name!r} at {path}:{line_number}")
        try:
            tvec = np.asarray(fields[1:4], dtype=float)
            qvec = np.asarray(fields[4:8], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Malformed anchor pose at {path}:{line_number}") from exc
        rotation = quaternion_to_rotation(qvec)
        poses[name] = AnchorPose(
            name=name,
            qvec=qvec,
            tvec=tvec,
            rotation=rotation,
            centre=-rotation.T @ tvec,
        )
        order.append(name)
    if not order:
        raise ValueError(f"No anchors found in {path}")
    return poses, order


def image_sort_key(name: str) -> tuple[Any, ...]:
    stem = Path(name).stem
    try:
        return (0, float(stem), name)
    except ValueError:
        tokens: list[tuple[int, Any]] = []
        for part in NATURAL_PARTS.split(name.lower()):
            if not part:
                continue
            tokens.append((1, int(part)) if part.isdigit() else (0, part))
        return (1, tuple(tokens), name)


def list_input_images(images_dir: Path) -> list[str]:
    images_dir = images_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {images_dir}")
    names = [
        path.relative_to(images_dir).as_posix()
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not names:
        raise ValueError(f"No supported images found under {images_dir}")
    return sorted(names, key=image_sort_key)


def build_segments(
    image_names: list[str],
    anchor_order: list[str],
    anchor_front: int,
    anchor_back: int,
) -> list[Segment]:
    stride = anchor_front + anchor_back
    if anchor_front <= 0 or anchor_back <= 0:
        raise ValueError("--anchor-front and --anchor-back must both be positive")
    if len(anchor_order) % stride != 0:
        raise ValueError(
            f"Anchor count {len(anchor_order)} is not divisible by front+back={stride}"
        )
    image_index = {name: index for index, name in enumerate(image_names)}
    segments: list[Segment] = []
    for segment_index, start in enumerate(range(0, len(anchor_order), stride)):
        front = tuple(anchor_order[start : start + anchor_front])
        back = tuple(anchor_order[start + anchor_front : start + stride])
        first_name, last_name = front[0], back[-1]
        fallback = first_name not in image_index or last_name not in image_index
        if not fallback:
            first_index, last_index = image_index[first_name], image_index[last_name]
            if first_index > last_index:
                raise ValueError(
                    f"Submap {segment_index} anchor order is reversed in --images-dir: "
                    f"{first_name} .. {last_name}"
                )
            segment_names = image_names[first_index : last_index + 1]
        else:
            first_key, last_key = image_sort_key(first_name), image_sort_key(last_name)
            if first_key > last_key:
                raise ValueError(
                    f"Submap {segment_index} anchor order is reversed: {first_name} .. {last_name}"
                )
            segment_names = [
                name for name in image_names if first_key <= image_sort_key(name) <= last_key
            ]
        segments.append(
            Segment(
                index=segment_index,
                front=front,
                back=back,
                image_names=tuple(segment_names),
                used_boundary_fallback=fallback,
            )
        )
    return segments


def pose_error(estimate: ImagePose | AnchorPose, ground_truth: ImagePose) -> tuple[float, float]:
    position = float(np.linalg.norm(estimate.centre - ground_truth.centre))
    cosine = float(
        np.clip(
            (np.trace(estimate.rotation @ ground_truth.rotation.T) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )
    return position, math.degrees(math.acos(cosine))


def distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "rmse": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "max": float(np.max(array)),
    }


def pair_error_summary(
    names: Iterable[str],
    estimates: dict[str, ImagePose] | dict[str, AnchorPose],
    ground_truth: dict[str, ImagePose],
) -> dict[str, Any]:
    rows = [pose_error(estimates[name], ground_truth[name]) for name in names if name in estimates and name in ground_truth]
    return {
        "matched": len(rows),
        "position_m": distribution(row[0] for row in rows),
        "rotation_deg": distribution(row[1] for row in rows),
    }


def analyse_track_topology(
    points_path: Path | None,
    id_to_name: dict[int, str],
    segments: list[Segment],
) -> tuple[list[dict[str, Any] | None], list[Counter]]:
    if points_path is None:
        return [None for _ in segments], [Counter() for _ in segments]
    category_by_name: dict[str, tuple[int, str]] = {}
    for segment in segments:
        front, back = set(segment.front), set(segment.back)
        for name in segment.image_names:
            category = "F" if name in front else "B" if name in back else "M"
            category_by_name[name] = (segment.index, category)
    track_counts = [Counter() for _ in segments]
    observation_counts = [Counter() for _ in segments]
    per_image_anchor_links = [Counter() for _ in segments]
    point_record_count = 0
    with points_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            point_record_count += 1
            fields = line.split()
            observations_by_segment: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for offset in range(8, len(fields), 2):
                try:
                    image_id = int(fields[offset])
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed track image id at {points_path}:{line_number}"
                    ) from exc
                name = id_to_name.get(image_id)
                if name in category_by_name:
                    segment_index, category = category_by_name[name]
                    observations_by_segment[segment_index].append((name, category))
            for segment_index, track_observations in observations_by_segment.items():
                categories = {category for _, category in track_observations}
                key = "".join(category for category in "FMB" if category in categories)
                track_counts[segment_index][key] += 1
                for name, category in track_observations:
                    observation_counts[segment_index][category] += 1
                    if category == "M":
                        if "F" in categories:
                            per_image_anchor_links[segment_index][(name, "F")] += 1
                        if "B" in categories:
                            per_image_anchor_links[segment_index][(name, "B")] += 1
    # A points-free pose export still contains a valid, header-only points3D.txt.
    # Treat that as unavailable topology rather than claiming that it measured
    # zero front/back bridge tracks.
    if point_record_count == 0:
        return [None for _ in segments], per_image_anchor_links
    results: list[dict[str, Any]] = []
    for segment_index in range(len(segments)):
        counts = track_counts[segment_index]
        observations = observation_counts[segment_index]
        results.append(
            {
                "tracks_total": int(sum(counts.values())),
                "tracks_front_only": int(counts["F"]),
                "tracks_middle_only": int(counts["M"]),
                "tracks_back_only": int(counts["B"]),
                "tracks_front_middle": int(counts["FM"]),
                "tracks_middle_back": int(counts["MB"]),
                "tracks_front_back": int(counts["FB"]),
                "tracks_front_middle_back": int(counts["FMB"]),
                "middle_to_back_bridge_tracks": int(counts["MB"] + counts["FMB"]),
                "track_observations_front": int(observations["F"]),
                "track_observations_middle": int(observations["M"]),
                "track_observations_back": int(observations["B"]),
            }
        )
    return results, per_image_anchor_links


def trajectory_scale_diagnostic(
    segment: Segment,
    estimates: dict[str, ImagePose],
    ground_truth: dict[str, ImagePose],
    min_observations: int,
    outlier_threshold_m: float,
) -> dict[str, Any]:
    anchor_names = set(segment.front + segment.back)
    good_names = [
        name
        for name in segment.image_names
        if name not in anchor_names
        and name in estimates
        and name in ground_truth
        and estimates[name].valid_observations >= min_observations
        and np.linalg.norm(estimates[name].centre - ground_truth[name].centre)
        < outlier_threshold_m
    ]
    result: dict[str, Any] = {
        "frames": len(good_names),
        "min_observations": min_observations,
        "outlier_threshold_m": outlier_threshold_m,
        "path_length_estimate_m": None,
        "path_length_gt_m": None,
        "path_length_ratio": None,
        "scale_about_last_front_anchor": None,
        "direction_cosine": None,
    }
    if len(good_names) < 2 or segment.front[-1] not in ground_truth:
        return result
    estimate_path = sum(
        float(np.linalg.norm(estimates[right].centre - estimates[left].centre))
        for left, right in zip(good_names, good_names[1:])
    )
    gt_path = sum(
        float(np.linalg.norm(ground_truth[right].centre - ground_truth[left].centre))
        for left, right in zip(good_names, good_names[1:])
    )
    origin = ground_truth[segment.front[-1]].centre
    gt_displacements = np.asarray([ground_truth[name].centre - origin for name in good_names])
    estimate_displacements = np.asarray([estimates[name].centre - origin for name in good_names])
    gt_energy = float(np.sum(gt_displacements * gt_displacements))
    estimate_energy = float(np.sum(estimate_displacements * estimate_displacements))
    cross = float(np.sum(gt_displacements * estimate_displacements))
    result["path_length_estimate_m"] = estimate_path
    result["path_length_gt_m"] = gt_path
    if gt_path > 0.0:
        result["path_length_ratio"] = estimate_path / gt_path
    if gt_energy > 0.0:
        result["scale_about_last_front_anchor"] = cross / gt_energy
    if gt_energy > 0.0 and estimate_energy > 0.0:
        result["direction_cosine"] = cross / math.sqrt(gt_energy * estimate_energy)
    return result


def evaluate_submap(
    segment: Segment,
    estimates: dict[str, ImagePose],
    ground_truth: dict[str, ImagePose],
    position_threshold_m: float,
    rotation_threshold_deg: float,
    scale_min_observations: int,
    scale_outlier_threshold_m: float,
    track_topology: dict[str, Any] | None,
) -> dict[str, Any]:
    anchors = set(segment.front + segment.back)
    frame_rows = []
    for name in segment.image_names:
        if name not in estimates or name not in ground_truth:
            continue
        position_error_m, rotation_error_deg = pose_error(estimates[name], ground_truth[name])
        frame_rows.append(
            {
                "name": name,
                "is_anchor": name in anchors,
                "position_error_m": position_error_m,
                "rotation_error_deg": rotation_error_deg,
                "observations": estimates[name].observations,
                "valid_observations": estimates[name].valid_observations,
            }
        )
    middle_rows = [row for row in frame_rows if not row["is_anchor"]]
    first_bad = next(
        (
            row
            for row in middle_rows
            if row["position_error_m"] > position_threshold_m
            or row["rotation_error_deg"] > rotation_threshold_deg
        ),
        None,
    )
    estimated_names = [name for name in segment.image_names if name in estimates]
    gt_names = [name for name in segment.image_names if name in ground_truth]
    estimated_anchors = [name for name in segment.front + segment.back if name in estimates]
    missing_estimates = [name for name in segment.image_names if name not in estimates]
    output_anchor_errors = pair_error_summary(
        segment.front + segment.back,
        estimates,
        ground_truth,
    )
    return {
        "submap": segment.index,
        "first_image": segment.image_names[0] if segment.image_names else None,
        "last_image": segment.image_names[-1] if segment.image_names else None,
        "boundary_fallback": segment.used_boundary_fallback,
        "frames": len(segment.image_names),
        "gt_frames": len(gt_names),
        "estimated_frames": len(estimated_names),
        "missing_estimates": len(missing_estimates),
        "missing_estimate_names": missing_estimates,
        "expected_anchors": len(segment.front) + len(segment.back),
        "estimated_anchors": len(estimated_anchors),
        "estimated_middle_frames": len(middle_rows),
        "position_error_m": distribution(row["position_error_m"] for row in middle_rows),
        "rotation_error_deg": distribution(row["rotation_error_deg"] for row in middle_rows),
        "output_anchor_error": output_anchor_errors,
        "first_threshold_crossing": first_bad,
        "trajectory_scale": trajectory_scale_diagnostic(
            segment,
            estimates,
            ground_truth,
            scale_min_observations,
            scale_outlier_threshold_m,
        ),
        "track_topology": track_topology,
        "frames_detail": frame_rows,
    }


def format_optional(value: float | int | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def format_distribution_triplet(stats: dict[str, Any]) -> str:
    if stats["count"] == 0:
        return "-"
    return "/".join(
        format_optional(stats[key], 3) for key in ("median", "mean", "max")
    )


def preview_names(names: list[str], limit: int) -> str:
    if not names:
        return ""
    preview = names[:limit]
    suffix = f" ... (+{len(names) - limit})" if len(names) > limit else ""
    return ", ".join(preview) + suffix


def print_summary(report: dict[str, Any], name_preview: int) -> None:
    counts = report["counts"]
    print(
        "Images: "
        f"input={counts['input_images']} GT={counts['gt_images']} "
        f"estimate={counts['estimated_images']} anchors={counts['anchors']} "
        f"submaps={counts['submaps']}"
    )
    for key, label in (
        ("input_not_gt", "input-GT"),
        ("gt_not_input", "GT-input"),
        ("input_not_estimate", "input-estimate"),
        ("estimate_not_input", "estimate-input"),
    ):
        names = report["name_differences"][key]
        if names:
            print(f"  {label}: {len(names)} [{preview_names(names, name_preview)}]")
    anchor_file = report["anchor_file_vs_gt"]
    output_anchor = report["output_anchor_vs_gt"]
    print(
        "Anchors: "
        f"file-vs-GT matched={anchor_file['matched']} "
        f"position mean/max={format_optional(anchor_file['position_m']['mean'], 8)}/"
        f"{format_optional(anchor_file['position_m']['max'], 8)} m, "
        f"output-vs-GT matched={output_anchor['matched']} "
        f"position mean/max={format_optional(output_anchor['position_m']['mean'], 8)}/"
        f"{format_optional(output_anchor['position_m']['max'], 8)} m"
    )
    print(
        "\n"
        "sm  frames  est/miss  anc  mid   centre med/mean/max (m)   "
        "rotation med/mean/max (deg)  scale  bridge  first > threshold"
    )
    for row in report["submaps"]:
        scale = row["trajectory_scale"]["scale_about_last_front_anchor"]
        topology = row["track_topology"]
        bridge = (
            str(topology["middle_to_back_bridge_tracks"])
            if topology is not None
            else "N/A"
        )
        first_bad = row["first_threshold_crossing"]
        if first_bad is None:
            first_text = "-"
        else:
            first_text = (
                f"{first_bad['name']} "
                f"({first_bad['position_error_m']:.3f}m/"
                f"{first_bad['rotation_error_deg']:.2f}deg,"
                f"obs={first_bad['valid_observations']})"
            )
        print(
            f"{row['submap']:2d}  {row['frames']:6d}  "
            f"{row['estimated_frames']:3d}/{row['missing_estimates']:<3d}  "
            f"{row['estimated_anchors']:2d}/{row['expected_anchors']:<2d} "
            f"{row['estimated_middle_frames']:4d}   "
            f"{format_distribution_triplet(row['position_error_m']):>24s}   "
            f"{format_distribution_triplet(row['rotation_error_deg']):>28s}  "
            f"{format_optional(scale, 3):>5s}  "
            f"{bridge:>6s}  {first_text}"
        )


def print_submap_detail(
    report: dict[str, Any],
    detail_submap: int,
    anchor_links: list[Counter],
) -> None:
    if detail_submap < 0:
        return
    if detail_submap >= len(report["submaps"]):
        raise ValueError(
            f"--detail-submap {detail_submap} is outside 0..{len(report['submaps']) - 1}"
        )
    row = report["submaps"][detail_submap]
    details_by_name = {frame["name"]: frame for frame in row["frames_detail"]}
    print(
        f"\nSubmap {detail_submap} frame detail:\n"
        "name role centre_m rotation_deg obs/valid tracks_to_front/back"
    )
    front = set(report["segments"][detail_submap]["front"])
    back = set(report["segments"][detail_submap]["back"])
    for name in report["segments"][detail_submap]["image_names"]:
        frame = details_by_name.get(name)
        if frame is None:
            print(f"{name} - missing/non-keyframe")
            continue
        role = "F" if name in front else "B" if name in back else "M"
        links = anchor_links[detail_submap]
        links_text = (
            f"{links[(name, 'F')]}/{links[(name, 'B')]}"
            if row["track_topology"] is not None
            else "N/A"
        )
        print(
            f"{name} {role} {frame['position_error_m']:.6f} "
            f"{frame['rotation_error_deg']:.4f} "
            f"{frame['observations']}/{frame['valid_observations']} "
            f"{links_text}"
        )


def csv_rows(report: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "submap",
        "first_image",
        "last_image",
        "frames",
        "gt_frames",
        "estimated_frames",
        "missing_estimates",
        "expected_anchors",
        "estimated_anchors",
        "estimated_middle_frames",
        "position_median_m",
        "position_mean_m",
        "position_p90_m",
        "position_rmse_m",
        "position_max_m",
        "rotation_median_deg",
        "rotation_mean_deg",
        "rotation_p90_deg",
        "rotation_rmse_deg",
        "rotation_max_deg",
        "first_bad_image",
        "first_bad_position_m",
        "first_bad_rotation_deg",
        "first_bad_valid_observations",
        "scale_frames",
        "path_length_ratio",
        "scale_about_last_front_anchor",
        "direction_cosine",
        "track_topology_available",
        "middle_to_back_bridge_tracks",
        "track_observations_front",
        "track_observations_middle",
        "track_observations_back",
    ]
    rows: list[dict[str, Any]] = []
    for submap in report["submaps"]:
        position = submap["position_error_m"]
        rotation = submap["rotation_error_deg"]
        first_bad = submap["first_threshold_crossing"] or {}
        scale = submap["trajectory_scale"]
        topology = submap["track_topology"] or {}
        rows.append(
            {
                "submap": submap["submap"],
                "first_image": submap["first_image"],
                "last_image": submap["last_image"],
                "frames": submap["frames"],
                "gt_frames": submap["gt_frames"],
                "estimated_frames": submap["estimated_frames"],
                "missing_estimates": submap["missing_estimates"],
                "expected_anchors": submap["expected_anchors"],
                "estimated_anchors": submap["estimated_anchors"],
                "estimated_middle_frames": submap["estimated_middle_frames"],
                "position_median_m": position["median"],
                "position_mean_m": position["mean"],
                "position_p90_m": position["p90"],
                "position_rmse_m": position["rmse"],
                "position_max_m": position["max"],
                "rotation_median_deg": rotation["median"],
                "rotation_mean_deg": rotation["mean"],
                "rotation_p90_deg": rotation["p90"],
                "rotation_rmse_deg": rotation["rmse"],
                "rotation_max_deg": rotation["max"],
                "first_bad_image": first_bad.get("name"),
                "first_bad_position_m": first_bad.get("position_error_m"),
                "first_bad_rotation_deg": first_bad.get("rotation_error_deg"),
                "first_bad_valid_observations": first_bad.get("valid_observations"),
                "scale_frames": scale["frames"],
                "path_length_ratio": scale["path_length_ratio"],
                "scale_about_last_front_anchor": scale["scale_about_last_front_anchor"],
                "direction_cosine": scale["direction_cosine"],
                "track_topology_available": submap["track_topology"] is not None,
                "middle_to_back_bridge_tracks": topology.get("middle_to_back_bridge_tracks"),
                "track_observations_front": topology.get("track_observations_front"),
                "track_observations_middle": topology.get("track_observations_middle"),
                "track_observations_back": topology.get("track_observations_back"),
            }
        )
    return fields, rows


def write_json(report: dict[str, Any], destination: str) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if destination == "-":
        print(payload)
        return
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")


def write_csv(report: dict[str, Any], destination: str) -> None:
    fields, rows = csv_rows(report)
    if destination == "-":
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strip_frame_details(report: dict[str, Any]) -> dict[str, Any]:
    """Keep JSON compact; frame details remain available in terminal detail mode."""
    compact = dict(report)
    compact["submaps"] = []
    for submap in report["submaps"]:
        row = dict(submap)
        row.pop("frames_detail", None)
        compact["submaps"].append(row)
    return compact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare anchor-submap COLMAP poses to GT by image name. "
            "GT/estimate may be a text-model directory or images.txt."
        )
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--est", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--anchor-front", type=int, required=True)
    parser.add_argument("--anchor-back", type=int, required=True)
    parser.add_argument("--position-threshold", type=float, default=0.5, metavar="METRES")
    parser.add_argument("--rotation-threshold", type=float, default=5.0, metavar="DEGREES")
    parser.add_argument(
        "--scale-min-observations",
        type=int,
        default=100,
        help=(
            "Only use estimated images with at least this many valid 3D observations "
            "for trajectory-scale diagnostics. Set 0 for points-free pose exports."
        ),
    )
    parser.add_argument("--scale-outlier-threshold", type=float, default=100.0, metavar="METRES")
    parser.add_argument("--detail-submap", type=int, default=-1)
    parser.add_argument("--name-preview", type=int, default=5)
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Optionally write a JSON report; use '-' for JSON-only stdout.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Optionally write one CSV row per submap; use '-' for CSV-only stdout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.position_threshold < 0.0 or args.rotation_threshold < 0.0:
        raise ValueError("Error thresholds must be non-negative")
    if args.scale_min_observations < 0 or args.scale_outlier_threshold <= 0.0:
        raise ValueError("Scale observation threshold must be >=0 and outlier threshold >0")
    if args.name_preview < 0:
        raise ValueError("--name-preview must be non-negative")
    if args.json == "-" and args.csv == "-":
        raise ValueError("Only one of --json/--csv may target stdout")

    gt_images_path, _ = resolve_model_paths(args.gt)
    estimate_images_path, estimate_points_path = resolve_model_paths(args.est)
    ground_truth, _ = read_colmap_images(gt_images_path)
    estimates, estimate_id_to_name = read_colmap_images(estimate_images_path)
    anchor_poses, anchor_order = read_anchors(args.anchors)
    input_images = list_input_images(args.images_dir)
    segments = build_segments(
        input_images,
        anchor_order,
        args.anchor_front,
        args.anchor_back,
    )
    track_topologies, anchor_links = analyse_track_topology(
        estimate_points_path,
        estimate_id_to_name,
        segments,
    )

    input_set, gt_set, estimate_set = set(input_images), set(ground_truth), set(estimates)
    submaps = [
        evaluate_submap(
            segment,
            estimates,
            ground_truth,
            args.position_threshold,
            args.rotation_threshold,
            args.scale_min_observations,
            args.scale_outlier_threshold,
            track_topologies[segment.index],
        )
        for segment in segments
    ]
    report = {
        "schema_version": 1,
        "configuration": {
            "images_dir": str(args.images_dir.expanduser().resolve()),
            "gt_images": str(gt_images_path),
            "estimate_images": str(estimate_images_path),
            "estimate_points3D": str(estimate_points_path) if estimate_points_path else None,
            "anchors": str(args.anchors.expanduser().resolve()),
            "anchor_front": args.anchor_front,
            "anchor_back": args.anchor_back,
            "position_threshold_m": args.position_threshold,
            "rotation_threshold_deg": args.rotation_threshold,
            "scale_min_observations": args.scale_min_observations,
            "scale_outlier_threshold_m": args.scale_outlier_threshold,
        },
        "counts": {
            "input_images": len(input_images),
            "gt_images": len(ground_truth),
            "estimated_images": len(estimates),
            "anchors": len(anchor_order),
            "submaps": len(segments),
        },
        "name_differences": {
            "input_not_gt": sorted(input_set - gt_set, key=image_sort_key),
            "gt_not_input": sorted(gt_set - input_set, key=image_sort_key),
            "input_not_estimate": sorted(input_set - estimate_set, key=image_sort_key),
            "estimate_not_input": sorted(estimate_set - input_set, key=image_sort_key),
        },
        "anchor_file_vs_gt": pair_error_summary(anchor_order, anchor_poses, ground_truth),
        "output_anchor_vs_gt": pair_error_summary(anchor_order, estimates, ground_truth),
        "segments": [
            {
                "submap": segment.index,
                "front": list(segment.front),
                "back": list(segment.back),
                "image_names": list(segment.image_names),
                "boundary_fallback": segment.used_boundary_fallback,
            }
            for segment in segments
        ],
        "submaps": submaps,
    }

    machine_stdout = args.json == "-" or args.csv == "-"
    if not machine_stdout:
        print_summary(report, args.name_preview)
        print_submap_detail(report, args.detail_submap, anchor_links)
    compact_report = strip_frame_details(report)
    if args.json:
        write_json(compact_report, args.json)
    if args.csv:
        write_csv(compact_report, args.csv)


if __name__ == "__main__":
    main()
