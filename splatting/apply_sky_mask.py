"""
sky mask RGBA
mask black=sky
    python tools/apply_sky_mask.py \
        --images data/staff_full/images \
        --masks  data/staff_full/mask \
        --output data/staff_full/images_masked
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_one(image_path: Path, mask_path: Path, output_path: Path, threshold: int, compress_level: int = 1):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)

    img_np = np.array(img)
    mask_np = np.array(mask)

    alpha = np.where(mask_np <= threshold, 0, 255).astype(np.uint8)

    rgba = np.dstack([img_np, alpha])
    Image.fromarray(rgba, "RGBA").save(output_path, format="PNG", compress_level=compress_level)


def main():
    parser = argparse.ArgumentParser(description="Apply sky mask to images (sky → transparent)")
    parser.add_argument("--images", required=True, help='Input image directory (search subdirectories recursively)')
    parser.add_argument("--masks", required=True, help='Sky-mask directory (black=sky, white=ground)')
    parser.add_argument("--output", required=True, help='Output directory (preserve input subdirectory structure)')
    parser.add_argument("--threshold", type=int, default=128,
                        help='Mask grayscale threshold; values <= threshold are sky (default: 128)')
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                        help='Worker process count (default: min(32, CPU count))')
    parser.add_argument("--compress_level", type=int, default=6,
                        help='PNG compression level 0-9; lower is faster with larger files (default: 1)')
    parser.add_argument("--ext", default=".png", help='Output extension (default: .png)')
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    image_files = sorted(
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in img_exts
    )

    if not image_files:
        print(f"[ERROR] No images found in {images_dir} (searched recursively)")
        sys.exit(1)

    tasks = []
    skipped = []
    for img_path in image_files:
        rel = img_path.relative_to(images_dir)
        mask_path = masks_dir / rel.with_suffix(".png")
        if not mask_path.exists():
            skipped.append(str(rel))
            continue
        out_path = output_dir / rel.with_suffix(args.ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((img_path, mask_path, out_path))

    if skipped:
        print(f"[WARNING] {len(skipped)} images have no matching mask and were skipped")

    print(f"[INFO] {len(tasks)} images to process -> {output_dir}  "
          f"(workers={args.workers}, compress_level={args.compress_level})")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_one, img, msk, out, args.threshold, args.compress_level)
            for img, msk, out in tasks
        ]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]")

    print(f"[DONE] {done} images saved to {output_dir}")


if __name__ == "__main__":
    main()
