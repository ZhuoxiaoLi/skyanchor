"""Export test-view lists that follow main ``train.py`` splitting rule.

Align with ``scene/dataset_readers.py:readColmapSceneInfo``:

1. Parse a COLMAP ``sparse/0/images.txt`` to collect every camera's
   ``image_name`` (basename without extension).
2. Drop entries whose image file cannot be found on disk, mirroring the
   silent-skip branch in ``readColmapCameras``.
3. Sort the surviving entries by ``image_name`` ascending.
4. Pick ``idx % llffhold == 0`` as the test set; the rest is train.

The script is intentionally standalone (only uses the stdlib) so it can be
shipped alongside datasets and re-run from any checkout.

Example
-------

    # Two independent scenes (aerial + ground) sharing the same park root.
    # Per-scene files are always written under each source_path.
    # --combined-output additionally writes a merged file for baselines
    # that load aerial + street/ground as one big COLMAP (e.g. HorizonGS).
    python tools/export_test_views.py \
        --source-path data/park/aerial \
        --source-path data/park/ground \
        --llffhold 8 \
        --combined-output baselines/HorizonGS/data/real/park/test_views.txt

Each ``--source-path`` produces ``<source_path>/test_views.txt`` with a
header block describing the split and two tab-separated columns
``image_name<TAB>colmap_name`` for every test camera. Downstream code
(e.g. HorizonGS) can then read the file and use either column to build an
identical held-out set.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple


def parse_images_txt(path: str) -> Dict[int, str]:
    """Return ``image_id -> colmap_name`` read from a COLMAP ``images.txt``.

    Mirrors the parsing logic of ``scene.colmap_loader.read_extrinsics_text``
    but keeps only the fields we actually need (id and name). The POINTS2D
    line that follows every metadata line is consumed even when empty, which
    matches how COLMAP writes the file.
    """
    id_to_name: Dict[int, str] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            elems = stripped.split()
            if len(elems) < 10:
                continue
            image_id = int(elems[0])
            name = elems[9]
            id_to_name[image_id] = name
            fid.readline()
    return id_to_name


def image_name_from_colmap_name(name: str) -> str:
    """``os.path.splitext(os.path.basename(name))[0]`` - matches main project."""
    return os.path.splitext(os.path.basename(name))[0]


def image_exists(colmap_name: str, images_folder: str) -> bool:
    """Replicate the two-step existence check used by ``readColmapCameras``.

    First probe ``images_folder/<colmap_name>`` verbatim; if it is missing,
    fall back to matching files in ``dirname(colmap_name)`` by basename
    without extension (so e.g. ``DSC06200.jpg`` will also match ``.png``
    on disk).
    """
    if not images_folder:
        return True
    direct = os.path.join(images_folder, colmap_name)
    if os.path.exists(direct):
        return True
    short = image_name_from_colmap_name(colmap_name)
    subdir = os.path.dirname(colmap_name)
    search_dir = os.path.join(images_folder, subdir) if subdir else images_folder
    if os.path.isdir(search_dir):
        for f in os.listdir(search_dir):
            if os.path.splitext(f)[0] == short:
                return True
    return False


class SceneResult:
    """Container for one scene's split outcome, used when building combined output."""

    __slots__ = ("source_path", "scene_name", "images_txt", "total",
                 "train_names", "test_names", "skipped", "verify_images")

    def __init__(
        self,
        source_path: str,
        scene_name: str,
        images_txt: str,
        total: int,
        train_names: List[str],
        test_names: List[str],
        skipped: List[str],
        verify_images: bool,
    ):
        self.source_path = source_path
        self.scene_name = scene_name
        self.images_txt = images_txt
        self.total = total
        self.train_names = train_names
        self.test_names = test_names
        self.skipped = skipped
        self.verify_images = verify_images


def export_scene(
    source_path: str,
    images_subdir: str,
    sparse_subdir: str,
    llffhold: int,
    output_path: str,
    verify_images: bool,
    scene_name: str = "",
) -> SceneResult:
    images_txt = os.path.join(source_path, sparse_subdir, "images.txt")
    if not os.path.exists(images_txt):
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_txt}")

    id_to_name = parse_images_txt(images_txt)
    images_folder = os.path.join(source_path, images_subdir)
    effective_verify = verify_images
    if effective_verify and not os.path.isdir(images_folder):
        print(f"  [warn] images folder not found: {images_folder}; skip existence check.")
        effective_verify = False

    kept: List[str] = []
    skipped: List[str] = []
    for colmap_name in id_to_name.values():
        if effective_verify and not image_exists(colmap_name, images_folder):
            skipped.append(colmap_name)
            continue
        kept.append(colmap_name)

    kept.sort(key=image_name_from_colmap_name)

    if llffhold <= 0:
        raise ValueError(f"llffhold must be positive, got {llffhold}")

    train_list: List[str] = []
    test_list: List[str] = []
    for idx, colmap_name in enumerate(kept):
        if idx % llffhold == 0:
            test_list.append(colmap_name)
        else:
            train_list.append(colmap_name)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Auto-generated by tools/export_test_views.py\n")
        f.write(f"# source_path:     {os.path.abspath(source_path)}\n")
        f.write(f"# images_txt:      {os.path.abspath(images_txt)}\n")
        f.write(f"# llffhold:        {llffhold}\n")
        f.write(f"# total_cameras:   {len(kept)}\n")
        f.write(f"# train_count:     {len(train_list)}\n")
        f.write(f"# test_count:      {len(test_list)}\n")
        f.write(f"# verify_images:   {effective_verify}\n")
        f.write(f"# skipped_missing: {len(skipped)}\n")
        f.write("# rule: sort cameras by image_name(no_ext) asc, "
                "then idx %% llffhold == 0 -> test\n")
        f.write("# columns: image_name<TAB>colmap_name\n")
        for colmap_name in test_list:
            short = image_name_from_colmap_name(colmap_name)
            f.write(f"{short}\t{colmap_name}\n")

    summary = (
        f"  total={len(kept)}  train={len(train_list)}  test={len(test_list)}"
    )
    if skipped:
        summary += f"  skipped_missing={len(skipped)}"
    print(summary)
    print(f"  -> {os.path.abspath(output_path)}")

    if not scene_name:
        scene_name = os.path.basename(os.path.abspath(source_path).rstrip(os.sep))
    return SceneResult(
        source_path=source_path,
        scene_name=scene_name,
        images_txt=images_txt,
        total=len(kept),
        train_names=train_list,
        test_names=test_list,
        skipped=skipped,
        verify_images=effective_verify,
    )


def write_combined_output(
    combined_output: str,
    results: List[SceneResult],
    llffhold: int,
) -> None:
    total_cams = sum(r.total for r in results)
    total_train = sum(len(r.train_names) for r in results)
    total_test = sum(len(r.test_names) for r in results)
    total_skipped = sum(len(r.skipped) for r in results)

    # Detect duplicate image_name across scenes so the user isn't surprised
    # by ambiguous basename matches downstream.
    short_to_scenes: Dict[str, List[str]] = {}
    for r in results:
        for colmap_name in r.test_names + r.train_names:
            short = image_name_from_colmap_name(colmap_name)
            short_to_scenes.setdefault(short, []).append(r.scene_name)
    duplicates = {k: v for k, v in short_to_scenes.items() if len(set(v)) > 1}

    out_dir = os.path.dirname(os.path.abspath(combined_output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(combined_output, "w") as f:
        f.write("# Auto-generated by tools/export_test_views.py (combined)\n")
        f.write(f"# llffhold:             {llffhold}\n")
        f.write(f"# scenes:               {len(results)}\n")
        for r in results:
            f.write(
                f"#   - {r.scene_name}: source_path={os.path.abspath(r.source_path)} "
                f"total={r.total} train={len(r.train_names)} test={len(r.test_names)} "
                f"skipped={len(r.skipped)} verify_images={r.verify_images}\n"
            )
        f.write(f"# total_cameras:        {total_cams}\n")
        f.write(f"# train_count:          {total_train}\n")
        f.write(f"# test_count:           {total_test}\n")
        f.write(f"# skipped_missing:      {total_skipped}\n")
        f.write(f"# duplicate_basenames:  {len(duplicates)}\n")
        if duplicates:
            f.write("# NOTE: the following image_name values appear in multiple scenes;\n")
            f.write("#       downstream code should disambiguate via the source_scene column.\n")
            for short, scenes in sorted(duplicates.items()):
                f.write(f"#   - {short}: {sorted(set(scenes))}\n")
        f.write("# rule: each scene is split independently "
                "(sort by image_name asc, idx %% llffhold == 0 -> test)\n")
        f.write("# columns: image_name<TAB>colmap_name<TAB>source_scene\n")
        for r in results:
            for colmap_name in r.test_names:
                short = image_name_from_colmap_name(colmap_name)
                f.write(f"{short}\t{colmap_name}\t{r.scene_name}\n")

    print("\n==== Combined output ====")
    print(
        f"  scenes={len(results)}  total={total_cams}  "
        f"train={total_train}  test={total_test}"
        + (f"  duplicate_basenames={len(duplicates)}" if duplicates else "")
    )
    print(f"  -> {os.path.abspath(combined_output)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export test-view lists by replaying main train.py's "
            "readColmapSceneInfo split rule on a COLMAP images.txt."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source-path",
        action="append",
        required=True,
        help="COLMAP source path (the folder containing sparse/0/images.txt).\n"
             "Pass multiple times to process several scenes in one call.",
    )
    parser.add_argument(
        "--scene-name",
        action="append",
        default=None,
        help="Optional name used as the source_scene column in --combined-output.\n"
             "Provide once per --source-path or omit to default to basename(source_path).",
    )
    parser.add_argument(
        "--images-subdir",
        default="images",
        help="Folder under source_path holding the actual images. Default: images",
    )
    parser.add_argument(
        "--sparse-subdir",
        default="sparse/0",
        help="Folder under source_path holding images.txt. Default: sparse/0",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="Hold-out stride, must be positive. Default: 8 (matches main train.py).",
    )
    parser.add_argument(
        "--output-name",
        default="test_views.txt",
        help="File name written inside each source_path. Default: test_views.txt",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Absolute output file path. Only valid when exactly one --source-path is given.",
    )
    parser.add_argument(
        "--combined-output",
        default=None,
        help="If given, also write a merged test_views.txt containing every scene's\n"
             "test cameras. Each line adds a third column source_scene so downstream\n"
             "baselines (e.g. HorizonGS) can locate the origin of each view.",
    )
    parser.add_argument(
        "--skip-per-scene",
        action="store_true",
        help="Do not write each scene's test_views.txt. Useful together with\n"
             "--combined-output when you only need the merged file.",
    )
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="Skip the on-disk existence check. Kept on by default so the exported "
             "list stays aligned with main train.py's silent-skip behaviour.",
    )
    args = parser.parse_args()

    if args.output_path and len(args.source_path) != 1:
        parser.error("--output-path can only be used together with a single --source-path")
    if args.output_path and args.skip_per_scene:
        parser.error("--output-path and --skip-per-scene cannot be combined")
    if args.skip_per_scene and not args.combined_output:
        parser.error("--skip-per-scene requires --combined-output")

    if args.scene_name is not None and len(args.scene_name) != len(args.source_path):
        parser.error(
            "--scene-name must be provided the same number of times as --source-path "
            f"(got {len(args.scene_name)} scene names for {len(args.source_path)} sources)"
        )

    results: List[SceneResult] = []
    for idx, source_path in enumerate(args.source_path):
        if not os.path.isdir(source_path):
            print(f"[skip] source_path does not exist: {source_path}", file=sys.stderr)
            continue
        print(f"\n==== Processing {source_path} ====")
        scene_name = args.scene_name[idx] if args.scene_name else ""

        if args.skip_per_scene:
            # Use a throw-away file so we can still exercise the single-scene writer
            # without polluting the user's dataset dirs.
            tmp_out = os.path.join(source_path, f".{args.output_name}.tmp")
            out = tmp_out
        else:
            out = args.output_path or os.path.join(source_path, args.output_name)

        res = export_scene(
            source_path=source_path,
            images_subdir=args.images_subdir,
            sparse_subdir=args.sparse_subdir,
            llffhold=args.llffhold,
            output_path=out,
            verify_images=not args.no_verify_images,
            scene_name=scene_name,
        )
        results.append(res)

        if args.skip_per_scene:
            try:
                os.remove(out)
            except OSError:
                pass

    if not results:
        sys.exit("No valid --source-path processed.")

    if args.combined_output:
        write_combined_output(
            combined_output=args.combined_output,
            results=results,
            llffhold=args.llffhold,
        )


if __name__ == "__main__":
    main()
