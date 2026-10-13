"""Validate the AG-to-pose contract and export ground anchors without loading 3D points."""

from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_names(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No images in {directory}")
    invalid = [
        p
        for p in files
        if p.parent != directory or p.suffix.lower() not in IMAGE_EXTENSIONS
    ]
    if invalid:
        raise ValueError(
            f"Ground images must be flat JPG/PNG files only (both stages must sort the same files): {invalid[:5]}"
        )
    names = [p.name for p in files]
    if any(any(c.isspace() for c in n) for n in names):
        raise ValueError(
            "Whitespace in image names is not supported by the anchor text format."
        )
    return names


def submap_ranges(count, size, front, back):
    """Match AGParser.collect_unregister_images, including the half-size tail merge."""
    if front < 2 or back < 1 or size <= front + back or count < front + back:
        raise ValueError(
            "Need front >= 2, back >= 1, submap size > front+back, and enough images."
        )
    if count <= size:
        ranges = [(0, count)]
    else:
        full, remainder = divmod(count, size)
        if remainder and remainder < size * 0.5:
            ranges = [(i * size, (i + 1) * size) for i in range(full - 1)]
            ranges.append(((full - 1) * size, count))
        else:
            ranges = [
                (i * size, min((i + 1) * size, count))
                for i in range(full + bool(remainder))
            ]
    if any(b - a <= front + back for a, b in ranges):
        raise ValueError(
            "Each submap needs separate front/back anchors and at least one free frame."
        )
    return ranges


def expected_anchors(names, ranges, front, back):
    return [
        name for a, b in ranges for name in names[a : a + front] + names[b - back : b]
    ]


def calibrated_intrinsics(path, names, image_dir, downsampling):
    import yaml
    import numpy as np
    from PIL import Image

    if not math.isfinite(downsampling) or downsampling <= 0:
        raise ValueError("Downsampling must be positive and finite.")
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or not config:
        raise ValueError("Ground intrinsics YAML must contain camera entries.")
    assignments = {}
    for camera in config.values():
        params = np.asarray(camera["params"], dtype=float)
        if (
            params.shape != (4,)
            or not np.isfinite(params).all()
            or min(params[:2]) <= 0
        ):
            raise ValueError(
                "Ground intrinsics must be finite PINHOLE [fx, fy, cx, cy]."
            )
        assigned = names if camera["images"] == "all" else camera["images"]
        if not isinstance(assigned, list):
            raise ValueError('Intrinsics images must be "all" or a list of names.')
        for name in assigned:
            if name not in names:
                continue
            if name in assignments:
                raise ValueError(f"Overlapping camera assignments: {name}")
            assignments[name] = params
    missing = set(names) - assignments.keys()
    if missing:
        raise ValueError(
            f"Intrinsics must cover the full ground sequence; missing {sorted(missing)[:8]}"
        )
    params = assignments[names[0]]
    if not all(
        np.allclose(p, params, rtol=1e-7, atol=1e-6) for p in assignments.values()
    ):
        raise ValueError(
            "anchorRecon currently requires shared intrinsics for all ground frames."
        )
    fx, fy, cx, cy = map(float, params)
    if not math.isclose(fx, fy, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError(
            "anchorRecon uses a single focal length; fx and fy must agree."
        )
    sizes = set()
    for name in names:
        with Image.open(Path(image_dir) / name) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError(f"Ground image sizes differ: {sizes}")
    width, height = sizes.pop()
    if min(round(width / downsampling), round(height / downsampling)) < 1:
        raise ValueError("Downsampling produces an empty image.")
    # cv2.resize pixel-center mapping: dst = (src + 0.5) / factor - 0.5.
    return {
        "original": [fx, fy, cx, cy],
        "width": width,
        "height": height,
        "downsampling": downsampling,
        "focal": fx / downsampling,
        "cx": (cx + 0.5) / downsampling - 0.5,
        "cy": (cy + 0.5) / downsampling - 0.5,
    }


def read_poses(model):
    sys.path.insert(0, str(ROOT / "anchorRecon"))
    from dataloaders.read_write_model import (
        read_images_binary,
        read_images_text,
        read_cameras_binary,
        read_cameras_text,
    )

    model = Path(model)
    if (model / "images.bin").is_file() and (model / "cameras.bin").is_file():
        return read_cameras_binary(model / "cameras.bin"), read_images_binary(
            model / "images.bin"
        )
    if (model / "images.txt").is_file() and (model / "cameras.txt").is_file():
        return read_cameras_text(model / "cameras.txt"), read_images_text(
            model / "images.txt"
        )
    raise FileNotFoundError(f"No cameras/images COLMAP model in {model}")


def export(
    model,
    image_dir,
    intrinsics,
    output,
    manifest,
    size=50,
    front=6,
    back=6,
    downsampling=2,
):
    import numpy as np

    names = image_names(image_dir)
    ranges = submap_ranges(len(names), size, front, back)
    wanted = expected_anchors(names, ranges, front, back)
    calibration = calibrated_intrinsics(intrinsics, names, image_dir, downsampling)
    cameras, images = read_poses(model)
    selected = {}
    ground_set = set(names)
    for image in images.values():
        if image.name not in ground_set:
            if Path(image.name).name in ground_set:
                raise ValueError(
                    f"Ambiguous relative image name {image.name}; use the same flat ground image root in both stages."
                )
            continue
        if image.name in selected:
            raise ValueError(f"Duplicate COLMAP image name: {image.name}")
        selected[image.name] = image
    missing = set(wanted) - selected.keys()
    if missing:
        raise ValueError(
            f"AG reconstruction is missing {len(missing)} required anchors: {sorted(missing)[:20]}"
        )
    # Export exactly the planned boundary anchors, never fix every ground frame in a full model.
    rows = ["# image_name tx ty tz qw qx qy qz"]
    for name in wanted:
        image = selected[name]
        camera = cameras[image.camera_id]
        params = np.asarray(camera.params)
        expected = calibration["original"]
        camera_params = (
            params
            if camera.model == "PINHOLE"
            else (
                [params[0], params[0], *params[1:]]
                if camera.model == "SIMPLE_PINHOLE"
                else []
            )
        )
        if (
            (camera.width, camera.height)
            != (calibration["width"], calibration["height"])
            or len(camera_params) != 4
            or not np.allclose(camera_params, expected, rtol=1e-6, atol=1e-4)
        ):
            raise ValueError(
                f"AG camera calibration disagrees with input images/YAML: {name}"
            )
        q = np.asarray(image.qvec, dtype=float)
        t = np.asarray(image.tvec, dtype=float)
        if (
            not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
        ):
            raise ValueError(f"Invalid anchor pose: {name}")
        rows.append(name + " " + " ".join(f"{v:.17g}" for v in [*t, *q]))
    # Check against the actual downstream inference code, without constructing a GPU dataset.
    from dataloaders.image_dataset import ImageDataset

    indices = [i for i, name in enumerate(names) if name in set(wanted)]
    inferred = ImageDataset._infer_anchor_submap_ranges(
        indices, len(names), front, back
    )
    if inferred != ranges:
        raise ValueError(
            f"Downstream submap inference differs: expected {ranges}, got {inferred}"
        )
    data = {
        "model": str(Path(model).resolve()),
        "image_dir": str(Path(image_dir).resolve()),
        "pose_convention": "world_to_camera",
        "quaternion_order": "wxyz",
        "image_names": names,
        "anchor_names": wanted,
        "submap_ranges": ranges,
        "anchor_front": front,
        "anchor_back": back,
        "intrinsics": calibration,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(rows) + "\n")
    Path(manifest).write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"Validated {len(wanted)} anchors / {len(names)} frames / {len(ranges)} submaps → {output}"
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submap-size", type=int, default=50)
    parser.add_argument("--anchor-front", type=int, default=6)
    parser.add_argument("--anchor-back", type=int, default=6)
    parser.add_argument("--downsampling", type=float, default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        names = image_names(args.images)
        ranges = submap_ranges(
            len(names), args.submap_size, args.anchor_front, args.anchor_back
        )
        calibration = calibrated_intrinsics(
            args.intrinsics, names, args.images, args.downsampling
        )
        print(
            f"Inputs valid: {len(names)} frames, submaps={ranges}, intrinsics={calibration}"
        )
        return
    export(
        args.model,
        args.images,
        args.intrinsics,
        args.output,
        args.manifest,
        args.submap_size,
        args.anchor_front,
        args.anchor_back,
        args.downsampling,
    )


if __name__ == "__main__":
    main()
