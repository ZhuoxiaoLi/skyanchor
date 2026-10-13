#!/usr/bin/env python3
"""
Extract matched camera poses from COLMAP images.txt by image filenames.

Default anchor format:
    # image_name tx ty tz qw qx qy qz
Optional: --anchors-with-fov adds hfov_deg vfov_deg (requires --cameras-txt).
--cameras-txt alone only writes pinhole_fov.txt, not FOV inside anchor files.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


PoseTuple = Tuple[float, float, float, float, float, float, float]
# base_name, pose, camera_id from images.txt (optional)
MatchedRow = Tuple[str, PoseTuple, Optional[int]]


def collect_image_names(image_dir: Path, recursive: bool = True) -> Set[str]:
    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = image_dir.rglob("*")
    else:
        pattern_iter = image_dir.glob("*")

    names: Set[str] = set()
    for p in pattern_iter:
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            names.add(p.name)
    return names


def load_image_names_from_file(names_file: Path) -> Set[str]:
    """
    Read a text file that lists one image name per line.
    Blank lines and lines starting with '#' are ignored.
    Each name is reduced to its basename (in case a relative path was written).
    """
    if not names_file.is_file():
        raise FileNotFoundError(f"names file not found: {names_file}")
    names: Set[str] = set()
    with names_file.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            names.add(Path(s).name)
    return names


def load_image_names_ordered(names_file: Path) -> List[str]:
    """
    Same as load_image_names_from_file but preserves file order
    and only keeps the first basename split by whitespace
    (so lines that already carry pose columns are tolerated).
    """
    if not names_file.is_file():
        raise FileNotFoundError(f"names file not found: {names_file}")
    out: List[str] = []
    seen: Set[str] = set()
    with names_file.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            base = Path(s.split()[0]).name
            if base in seen:
                continue
            seen.add(base)
            out.append(base)
    return out


def parse_frame_number(name: str, pattern: re.Pattern) -> Optional[int]:
    m = pattern.search(name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (IndexError, ValueError):
        return None


def detect_clusters_by_frame_gap(
    indexed_names: List[Tuple[str, int]],
    gap_threshold: int,
) -> List[List[int]]:
    """
    indexed_names: list of (basename, frame_number) sorted by frame_number.
    Returns clusters as lists of positions in indexed_names.
    Two consecutive entries belong to different clusters if their frame_number
    difference is strictly greater than gap_threshold.
    """
    clusters: List[List[int]] = []
    if not indexed_names:
        return clusters
    current = [0]
    for i in range(1, len(indexed_names)):
        if indexed_names[i][1] - indexed_names[i - 1][1] > gap_threshold:
            clusters.append(current)
            current = [i]
        else:
            current.append(i)
    clusters.append(current)
    return clusters


def trim_cluster_edges(positions: List[int], edge_n: int) -> List[int]:
    """Keep the first edge_n and the last edge_n positions, deduped and ordered."""
    if edge_n <= 0 or len(positions) <= 2 * edge_n:
        return list(positions)
    head = positions[:edge_n]
    tail = positions[-edge_n:]
    out: List[int] = list(head)
    seen: Set[int] = set(head)
    for idx in tail:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def parse_colmap_images_txt(images_txt: Path) -> Tuple[Dict[str, PoseTuple], Dict[str, int]]:
    """
    Parse COLMAP images.txt format.
    Valid image record line format:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    followed by one line of 2D points.
    NAME may contain spaces; everything after CAMERA_ID is the image name.
    """
    poses: Dict[str, PoseTuple] = {}
    camera_ids: Dict[str, int] = {}
    with images_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            # IMAGE_ID must be int; this helps skip 2D points lines.
            try:
                int(parts[0])
            except ValueError:
                continue

            # Parse pose values.
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
            except ValueError:
                continue

            image_name = " ".join(parts[9:]) if len(parts) > 9 else parts[9]
            poses[image_name] = (tx, ty, tz, qw, qx, qy, qz)
            camera_ids[image_name] = camera_id
    return poses, camera_ids


def write_anchor_txt(
    output_txt: Path,
    ordered_matches: Iterable[MatchedRow],
    fov_by_cam: Optional[Dict[int, Tuple[float, float]]] = None,
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        if fov_by_cam is not None:
            f.write("# image_name tx ty tz qw qx qy qz hfov_deg vfov_deg\n")
            for image_name, (tx, ty, tz, qw, qx, qy, qz), cam_id in ordered_matches:
                if cam_id is not None and cam_id in fov_by_cam:
                    hfov, vfov = fov_by_cam[cam_id]
                    f.write(
                        f"{image_name} "
                        f"{tx:.8f} {ty:.8f} {tz:.8f} "
                        f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                        f"{hfov:.8f} {vfov:.8f}\n"
                    )
                else:
                    f.write(
                        f"{image_name} "
                        f"{tx:.8f} {ty:.8f} {tz:.8f} "
                        f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                        f"nan nan\n"
                    )
        else:
            f.write("# image_name tx ty tz qw qx qy qz\n")
            for image_name, (tx, ty, tz, qw, qx, qy, qz), _cam_id in ordered_matches:
                f.write(
                    f"{image_name} "
                    f"{tx:.8f} {ty:.8f} {tz:.8f} "
                    f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}\n"
                )


def edge_indices(chunk_len: int, edge_n: int) -> List[int]:
    """First edge_n and last edge_n indices; dedupe preserving order (front then back)."""
    if chunk_len <= 0:
        return []
    n = min(edge_n, chunk_len)
    first = list(range(n))
    last = list(range(chunk_len - n, chunk_len))
    out: List[int] = []
    seen: Set[int] = set()
    for i in first + last:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def submap_edge_chunks(
    matched_sorted: List[MatchedRow],
    submap_size: int,
    edge_n: int,
) -> List[List[MatchedRow]]:
    """
    Split matched_sorted into consecutive chunks of length submap_size (last chunk may be shorter).
    Per chunk keep only first edge_n and last edge_n rows (by chunk order), then sort by image name.
    """
    chunks: List[List[Tuple[str, PoseTuple]]] = []
    for start in range(0, len(matched_sorted), submap_size):
        chunk = matched_sorted[start : start + submap_size]
        idxs = edge_indices(len(chunk), edge_n)
        edge_rows = [chunk[j] for j in idxs]
        edge_rows.sort(key=lambda x: x[0])
        chunks.append(edge_rows)
    return chunks


def merge_submap_edge_rows(chunks: List[List[MatchedRow]]) -> List[MatchedRow]:
    """Flatten all submap edge rows and sort by image name (global order)."""
    merged: List[MatchedRow] = [row for chunk in chunks for row in chunk]
    merged.sort(key=lambda x: x[0])
    return merged


CameraPinholeRow = Tuple[int, int, int, float, float, float, float, float, float]


def pinhole_fov_degrees(width: int, height: int, fx: float, fy: float) -> Tuple[float, float]:
    """Horizontal and vertical field of view in degrees (COLMAP PINHOLE)."""
    hfov = 2.0 * math.degrees(math.atan2(float(width), 2.0 * fx))
    vfov = 2.0 * math.degrees(math.atan2(float(height), 2.0 * fy))
    return hfov, vfov


def parse_colmap_cameras_txt(
    cameras_txt: Path,
) -> Tuple[List[CameraPinholeRow], Dict[int, Tuple[float, float]]]:
    """
    Parse COLMAP cameras.txt. Supported models for FOV: PINHOLE, SIMPLE_PINHOLE, OPENCV
    (same fx/fy interpretation for diagonal pinhole FOV).
    Returns rows for pinhole_fov.txt and cam_id -> (hfov_deg, vfov_deg) for anchor joining.
    """
    rows: List[CameraPinholeRow] = []
    fov_by_cam: Dict[int, Tuple[float, float]] = {}
    with cameras_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            model = parts[1].upper()
            try:
                cam_id = int(parts[0])
                w, h = int(parts[2]), int(parts[3])
            except (ValueError, IndexError):
                continue

            fx: float
            fy: float
            cx: float
            cy: float
            if model == "PINHOLE" and len(parts) >= 9:
                try:
                    fx, fy = float(parts[4]), float(parts[5])
                    cx, cy = float(parts[6]), float(parts[7])
                except (ValueError, IndexError):
                    continue
            elif model == "SIMPLE_PINHOLE" and len(parts) >= 8:
                try:
                    f_len = float(parts[4])
                    cx, cy = float(parts[5]), float(parts[6])
                    fx = fy = f_len
                except (ValueError, IndexError):
                    continue
            elif model == "OPENCV" and len(parts) >= 9:
                try:
                    fx, fy = float(parts[4]), float(parts[5])
                    cx, cy = float(parts[6]), float(parts[7])
                except (ValueError, IndexError):
                    continue
            else:
                continue

            if fx <= 0 or fy <= 0:
                continue
            hfov, vfov = pinhole_fov_degrees(w, h, fx, fy)
            rows.append((cam_id, w, h, fx, fy, cx, cy, hfov, vfov))
            fov_by_cam[cam_id] = (hfov, vfov)
    rows.sort(key=lambda r: r[0])
    return rows, fov_by_cam


def write_pinhole_fov_txt(output_path: Path, rows: List[CameraPinholeRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# COLMAP PINHOLE camera FOV (degrees)\n")
        f.write(
            "# camera_id width height fx fy cx cy hfov_deg vfov_deg\n"
        )
        for (
            cam_id,
            w,
            h,
            fx,
            fy,
            cx,
            cy,
            hfov,
            vfov,
        ) in rows:
            f.write(
                f"{cam_id} {w} {h} "
                f"{fx:.9f} {fy:.9f} {cx:.9f} {cy:.9f} "
                f"{hfov:.8f} {vfov:.8f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match COLMAP images.txt records against image filenames and export sfm anchors txt."
    )
    parser.add_argument(
        "--colmap-txt",
        required=True,
        type=Path,
        help="Path to COLMAP images.txt (or same-format colmap txt).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing images to match by filename. "
        "Provide --image-dir and/or --names-file (at least one).",
    )
    parser.add_argument(
        "--names-file",
        type=Path,
        default=None,
        help="Text file listing image names (one per line, basenames). "
        "Can be used instead of or together with --image-dir. "
        "If both are given, the union of names is used.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output txt path, e.g. sfm_anchors.txt",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan top-level files in image-dir (default: recursive).",
    )
    parser.add_argument(
        "--cameras-txt",
        type=Path,
        default=None,
        help="COLMAP cameras.txt: write pinhole_fov.txt (PINHOLE / SIMPLE_PINHOLE / OPENCV). "
        "Does not add FOV columns to anchor files unless --anchors-with-fov.",
    )
    parser.add_argument(
        "--anchors-with-fov",
        action="store_true",
        help="Append hfov_deg vfov_deg to anchor outputs (requires --cameras-txt).",
    )
    parser.add_argument(
        "--fov-output-dir",
        type=Path,
        default=None,
        help="Directory for pinhole_fov.txt (default: same directory as --output).",
    )
    parser.add_argument(
        "--submap-size",
        type=int,
        default=None,
        metavar="N",
        help="Split image-name-sorted anchors into consecutive submaps of N images; last submap may be shorter.",
    )
    parser.add_argument(
        "--submap-edge-n",
        type=int,
        default=None,
        metavar="N",
        help="Per submap keep first N and last N anchors (requires --submap-size and --submap-output).",
    )
    parser.add_argument(
        "--submap-output",
        type=Path,
        default=None,
        help="Single txt path: all submap edge anchors merged, sorted by image name (requires --submap-size and --submap-edge-n).",
    )
    parser.add_argument(
        "--anchors-list",
        type=Path,
        default=None,
        help="List-driven mode: a text file with one image basename per line. "
        "The script auto-detects clusters from frame-number gaps in the list, "
        "optionally trims each cluster to first/last N entries, looks up poses "
        "from --colmap-txt, and writes the result to --output as submap anchors. "
        "When this flag is set, --image-dir / --names-file are not required.",
    )
    parser.add_argument(
        "--cluster-edge-n",
        type=int,
        default=0,
        metavar="N",
        help="In --anchors-list mode, per detected cluster keep first N + last N entries (deduped). "
        "0 keeps every entry of every cluster.",
    )
    parser.add_argument(
        "--cluster-gap-threshold",
        type=int,
        default=50,
        metavar="T",
        help="In --anchors-list mode, two consecutive anchors with frame-number gap > T "
        "start a new cluster.",
    )
    parser.add_argument(
        "--frame-number-regex",
        type=str,
        default=r"(\d+)(?=\D*$)",
        help="Regex used to parse the frame number from the image basename. "
        "Default grabs the LAST run of digits before the extension.",
    )
    parser.add_argument(
        "--allow-missing-anchors",
        action="store_true",
        help="In --anchors-list mode, do not error when a listed name is missing in --colmap-txt; "
        "skip it with a warning instead.",
    )
    args = parser.parse_args()

    if args.submap_size is not None:
        if args.submap_size < 1:
            raise ValueError("--submap-size must be >= 1")
        if args.submap_edge_n is None or args.submap_edge_n < 1:
            raise ValueError("--submap-edge-n is required and must be >= 1 when --submap-size is set")
        if args.submap_output is None:
            raise ValueError("--submap-output is required when --submap-size is set")
    else:
        if args.submap_edge_n is not None or args.submap_output is not None:
            raise ValueError("--submap-edge-n / --submap-output require --submap-size")

    if args.anchors_with_fov and args.cameras_txt is None:
        raise ValueError("--anchors-with-fov requires --cameras-txt")

    if not args.colmap_txt.is_file():
        raise FileNotFoundError(f"COLMAP txt not found: {args.colmap_txt}")

    list_mode = args.anchors_list is not None
    if list_mode:
        if args.submap_size is not None:
            raise ValueError(
                "--anchors-list does its own cluster splitting; "
                "do not combine it with --submap-size / --submap-edge-n / --submap-output."
            )
        if args.cluster_edge_n < 0:
            raise ValueError("--cluster-edge-n must be >= 0")
        if args.cluster_gap_threshold < 1:
            raise ValueError("--cluster-gap-threshold must be >= 1")
    else:
        if args.image_dir is None and args.names_file is None:
            raise ValueError(
                "at least one of --image-dir / --names-file / --anchors-list must be provided"
            )
        if args.image_dir is not None and not args.image_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {args.image_dir}")

    poses_by_name, camera_by_name = parse_colmap_images_txt(args.colmap_txt)

    fov_by_cam: Optional[Dict[int, Tuple[float, float]]] = None
    pinhole_rows: List[CameraPinholeRow] = []
    if args.cameras_txt is not None:
        if not args.cameras_txt.is_file():
            raise FileNotFoundError(f"cameras.txt not found: {args.cameras_txt}")
        pinhole_rows, fov_by_cam = parse_colmap_cameras_txt(args.cameras_txt)
        if not pinhole_rows:
            print(
                'Warning: cameras.txt contains no supported camera models '
                '(PINHOLE / SIMPLE_PINHOLE / OPENCV); pinhole_fov.txt may contain only its header; '
                'anchor hfov/vfov values will be nan.'
            )

    anchor_fov: Optional[Dict[int, Tuple[float, float]]] = (
        fov_by_cam if (args.anchors_with_fov and fov_by_cam is not None) else None
    )

    if list_mode:
        try:
            frame_pattern = re.compile(args.frame_number_regex)
        except re.error as exc:
            raise ValueError(
                f"invalid --frame-number-regex {args.frame_number_regex!r}: {exc}"
            )

        listed_names = load_image_names_ordered(args.anchors_list)
        if not listed_names:
            raise ValueError(f"--anchors-list {args.anchors_list} contains no entries")

        # Resolve each listed name against COLMAP poses, allowing relative paths.
        poses_by_basename: Dict[str, Tuple[PoseTuple, Optional[int]]] = {}
        poses_by_basename_lower: Dict[str, Tuple[PoseTuple, Optional[int]]] = {}
        for raw_name, pose in poses_by_name.items():
            base = Path(raw_name).name
            poses_by_basename.setdefault(base, (pose, camera_by_name.get(raw_name)))
            poses_by_basename_lower.setdefault(
                base.lower(), (pose, camera_by_name.get(raw_name))
            )

        matched_rows: List[MatchedRow] = []
        unparsed: List[str] = []
        missing: List[str] = []
        for name in listed_names:
            frame_no = parse_frame_number(name, frame_pattern)
            if frame_no is None:
                unparsed.append(name)
                continue
            entry = poses_by_basename.get(name) or poses_by_basename_lower.get(name.lower())
            if entry is None:
                missing.append(name)
                continue
            pose, cam_id = entry
            matched_rows.append((name, pose, cam_id))

        if unparsed:
            preview = ", ".join(unparsed[:5])
            raise ValueError(
                f"--anchors-list contains {len(unparsed)} name(s) that do not match "
                f"--frame-number-regex {args.frame_number_regex!r}, e.g. {preview}. "
                f"Adjust the regex or clean the list."
            )

        if missing and not args.allow_missing_anchors:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"{len(missing)} listed name(s) are not present in --colmap-txt: {preview}. "
                f"Re-run with --allow-missing-anchors to skip them."
            )

        # Sort by frame number; detect clusters; optionally trim to first/last N.
        matched_rows.sort(key=lambda row: parse_frame_number(row[0], frame_pattern) or 0)
        indexed = [
            (row[0], parse_frame_number(row[0], frame_pattern) or 0)
            for row in matched_rows
        ]
        clusters = detect_clusters_by_frame_gap(indexed, args.cluster_gap_threshold)

        cluster_summaries: List[str] = []
        kept_rows: List[MatchedRow] = []
        edge_n = args.cluster_edge_n
        for cluster in clusters:
            kept = trim_cluster_edges(cluster, edge_n)
            kept_rows.extend(matched_rows[i] for i in kept)
            first_frame = indexed[cluster[0]][1]
            last_frame = indexed[cluster[-1]][1]
            cluster_summaries.append(
                f"[{first_frame}..{last_frame}] full={len(cluster)}, kept={len(kept)}"
            )

        write_anchor_txt(args.output, kept_rows, anchor_fov)

        print(f"anchors-list entries: {len(listed_names)}")
        print(f"poses in colmap: {len(poses_by_name)}")
        print(f"matched poses: {len(matched_rows)}")
        print(f"detected clusters: {len(clusters)} (gap_threshold={args.cluster_gap_threshold})")
        for summary in cluster_summaries:
            print(f"  cluster {summary}")
        print(
            f"cluster_edge_n={edge_n} -> total kept rows: {len(kept_rows)}"
        )
        print(f"saved to: {args.output}")
        if missing:
            preview = ", ".join(missing[:5])
            print(
                f"warning: skipped {len(missing)} listed name(s) absent from COLMAP, e.g. {preview}"
            )

        if args.cameras_txt is not None:
            fov_dir: Path = (
                args.fov_output_dir if args.fov_output_dir is not None else args.output.parent
            )
            fov_dir.mkdir(parents=True, exist_ok=True)
            fov_path = fov_dir / "pinhole_fov.txt"
            write_pinhole_fov_txt(fov_path, pinhole_rows)
            print(f"cameras parsed for FOV file: {len(pinhole_rows)}")
            print(f"FOV saved to: {fov_path}")
        return

    image_names: Set[str] = set()
    if args.image_dir is not None:
        image_names |= collect_image_names(args.image_dir, recursive=not args.non_recursive)
    if args.names_file is not None:
        image_names |= load_image_names_from_file(args.names_file)

    image_names_lower = {n.lower() for n in image_names}

    matched: List[MatchedRow] = []
    for name, pose in poses_by_name.items():
        # COLMAP name may contain relative folders like "xxx/yyy.jpg".
        base_name = Path(name).name
        if name in image_names or base_name in image_names or name.lower() in image_names_lower or base_name.lower() in image_names_lower:
            matched.append((base_name, pose, camera_by_name.get(name)))

    # Sort by image filename in ascending order.
    matched.sort(key=lambda x: x[0])

    write_anchor_txt(args.output, matched, anchor_fov)

    print(f"image names collected: {len(image_names)}")
    print(f"poses in colmap: {len(poses_by_name)}")
    print(f"matched poses: {len(matched)}")
    print(f"saved to: {args.output}")

    if args.names_file is not None and len(image_names) > 0:
        matched_names_lower = {row[0].lower() for row in matched}
        missing = [n for n in sorted(image_names) if n.lower() not in matched_names_lower]
        if missing:
            print(
                f"warning: {len(missing)} name(s) from input were NOT found in "
                f"COLMAP images.txt, e.g. {missing[:5]}"
            )

    if args.submap_size is not None:
        assert args.submap_edge_n is not None and args.submap_output is not None
        chunks = submap_edge_chunks(matched, args.submap_size, args.submap_edge_n)
        merged_edges = merge_submap_edge_rows(chunks)
        write_anchor_txt(args.submap_output, merged_edges, anchor_fov)
        total_edge = len(merged_edges)
        print(
            f"submaps: {len(chunks)} (size={args.submap_size}, edge_n={args.submap_edge_n}), "
            f"edge rows total: {total_edge}"
        )
        print(f"submap edge anchors (single file) saved to: {args.submap_output}")

    if args.cameras_txt is not None:
        fov_dir: Path = args.fov_output_dir if args.fov_output_dir is not None else args.output.parent
        fov_dir.mkdir(parents=True, exist_ok=True)
        fov_path = fov_dir / "pinhole_fov.txt"
        write_pinhole_fov_txt(fov_path, pinhole_rows)
        print(f"cameras parsed for FOV file: {len(pinhole_rows)}")
        print(f"FOV saved to: {fov_path}")


if __name__ == "__main__":
    main()
