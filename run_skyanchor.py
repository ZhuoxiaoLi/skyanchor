#!/usr/bin/env python3
"""Run aerial-ground registration, ground pose reconstruction, and Gaussian Splatting."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STD = ROOT.parent / "on-the-fly-nvs/data/isprs_std"


def interpreter(value):
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"Python interpreter not found: {value}")
    return str(Path(resolved).absolute())


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--ground-images", type=Path, default=STD / "ground/images")
    p.add_argument("--aerial-images", type=Path, default=STD / "images-aerial")
    p.add_argument("--aerial-model", type=Path, default=STD / "sparse-aerial")
    p.add_argument("--intrinsics", type=Path, default=STD / "intrinsics.yaml")
    p.add_argument(
        "--output",
        type=Path,
        help="New output directory; default outputs/<timestamp>. Existing nonempty directories are rejected.",
    )
    default_python = sys.executable
    p.add_argument("--ag-python", default=os.environ.get("AG_PYTHON", default_python))
    p.add_argument(
        "--pose-python", default=os.environ.get("POSE_PYTHON", default_python)
    )
    p.add_argument("--ag-config", default="sp-mast3r-dense")
    p.add_argument(
        "--ag-model",
        type=Path,
        help="Reuse an existing AG COLMAP model and skip AG reconstruction.",
    )
    p.add_argument("--submap-size", type=int, default=50)
    p.add_argument("--anchor-front", type=int, default=6)
    p.add_argument("--anchor-back", type=int, default=6)
    p.add_argument("--downsampling", type=float, default=2)
    p.add_argument("--scene-scale", type=float, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-keypoints", type=int, default=4500)
    p.add_argument("--viewer-port", type=int, default=8090)
    p.add_argument(
        "--headless", action="store_true", help="Disable viser and exit after export."
    )
    p.add_argument(
        "--exit-after-export",
        action="store_true",
        help="Enable viser during reconstruction, then exit.",
    )
    p.add_argument(
        "--gt-model",
        type=Path,
        help="Optional GT overlay; omit for scenes with no ground truth.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all stage commands without launching jobs or creating outputs.",
    )
    p.add_argument('--pose-only', action='store_true', help='Stop after pose reconstruction; skip splatting')
    p.add_argument('--splatting-only', type=Path, help='Reuse a completed SkyAnchor run and train splatting only')
    p.add_argument('--splatting-prepare-only', action='store_true', help='Prepare training data and sky masks without optimization')
    p.add_argument('--splatting-python', default=os.environ.get('SPLATTING_PYTHON', default_python))
    p.add_argument('--aerial-stage-iterations', type=int, default=30000)
    p.add_argument('--ground-stage-iterations', type=int, default=30000)
    p.add_argument('--joint-finetune-iterations', type=int, default=40000)
    p.add_argument('--sky-model', type=Path, default=ROOT/'splatting/submodules/SkyRemoval/model/model.onnx')
    p.add_argument('--skip-sky-removal', action='store_true')
    p.add_argument('--splatting-init', choices=['dense', 'sparse'], default='dense', help='RoMa dense initialization or exported SfM points')
    a = p.parse_args(argv)
    if a.pose_only and a.splatting_only:
        p.error('--pose-only and --splatting-only are mutually exclusive')
    if min(a.aerial_stage_iterations, a.ground_stage_iterations) < 1 or a.joint_finetune_iterations < 0:
        p.error('Aerial/ground iteration counts must be positive; joint must be nonnegative')
    a.splatting_python = interpreter(a.splatting_python)
    a.sky_model = a.sky_model.expanduser().resolve()
    if a.splatting_only:
        a.splatting_only = a.splatting_only.expanduser().resolve()

    if (
        a.anchor_front < 2
        or a.anchor_back < 1
        or a.submap_size <= a.anchor_front + a.anchor_back
    ):
        p.error("Require front >= 2, back >= 1, submap-size > front+back.")
    import math

    if (
        not all(math.isfinite(v) and v > 0 for v in [a.downsampling, a.scene_scale])
        or a.num_keypoints < 1
    ):
        p.error(
            "Downsampling, scene-scale and num-keypoints must be positive and finite."
        )
    if not 1 <= a.viewer_port <= 65535:
        p.error("viewer-port must be between 1 and 65535.")
    for key in [
        "ground_images",
        "aerial_images",
        "aerial_model",
        "intrinsics",
        "ag_model",
        "gt_model",
    ]:
        value = getattr(a, key)
        if value is not None:
            setattr(a, key, value.expanduser().resolve())
    a.output = (
        (a.output or ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        .expanduser()
        .resolve()
    )
    a.ag_python = interpreter(a.ag_python)
    a.pose_python = interpreter(a.pose_python)
    return a


def commands(a, calibration=None):
    ag_dir, pose_dir = a.output / "ag", a.output / "ground"
    model = a.ag_model or ag_dir / "sfm_outputs/rec"
    bridge = [
        a.pose_python,
        str(ROOT / "tools/export_ag_anchors.py"),
        "--model",
        str(model),
        "--images",
        str(a.ground_images),
        "--intrinsics",
        str(a.intrinsics),
        "--output",
        str(a.output / "submap_anchors.txt"),
        "--manifest",
        str(a.output / "handoff.json"),
        "--submap-size",
        str(a.submap_size),
        "--anchor-front",
        str(a.anchor_front),
        "--anchor-back",
        str(a.anchor_back),
        "--downsampling",
        str(a.downsampling),
    ]
    stages = [("01_validate", ROOT, bridge + ["--validate-only"])]
    if not a.ag_model:
        stages.append(
            (
                "02_ag",
                ROOT / "agRecon",
                [
                    a.ag_python,
                    "-u",
                    "reconstruct_ag.py",
                    "--rgb_register_dir",
                    str(a.aerial_images),
                    "--rgb_unregister_dir",
                    str(a.ground_images),
                    "--recon_exist_dir",
                    str(a.aerial_model),
                    "--intrinsics_pth",
                    str(a.intrinsics),
                    "--data_output_dir",
                    str(ag_dir),
                    "--cache_dir",
                    str(ag_dir / "cache_dir"),
                    "--conf",
                    a.ag_config,
                    "--submap_im_num",
                    str(a.submap_size),
                    "--anchor_num_pair",
                    str(a.anchor_front),
                    str(a.anchor_back),
                    "--verbose",
                    "0",
                ],
            )
        )
    stages.append(("03_export", ROOT, bridge))
    cal = calibration or {
        "focal": "<from-intrinsics>",
        "cx": "<scaled-cx>",
        "cy": "<scaled-cy>",
    }
    pose = [
        a.pose_python,
        "-u",
        "train.py",
        "-s",
        str(a.ground_images.parent),
        "-i",
        a.ground_images.name,
        "-m",
        str(pose_dir),
    ]
    values = {
        "viewer_mode": "none",
        "pose_debug_port": a.viewer_port,
        "random_seed": a.seed,
        "anchors_path": a.output / "submap_anchors.txt",
        "anchor_front": a.anchor_front,
        "anchor_back": a.anchor_back,
        "anchor_submap_size": 0,
        "bootstrap_seed_init_mode": "anchor_pnp",
        "matcher_backend": "lomag",
        "loma_num_keypoints": a.num_keypoints,
        "loma_magsac_threshold": 1,
        "loma_magsac_confidence": 0.999999,
        "loma_magsac_max_iters": 50000,
        "loma_filter_threshold": 0.01,
        "match_max_error": 0.001,
        "min_num_inliers": 30,
        "num_keyframes_miniba_bootstrap": max(8, a.anchor_front + 2),
        "num_pts_miniba_bootstrap": 6000,
        "iters_miniba_bootstrap": 350,
        "num_prev_keyframes_miniba_incr": 4,
        "num_prev_keyframes_miniba_incr_retry": 6,
        "num_prev_keyframes_check": 6,
        "pnpransac_samples": 8000,
        "num_pts_miniba_incr": 6000,
        "iters_miniba_incr": 300,
        "online_anchor_ba_every": 10,
        "online_anchor_ba_num_pts": 6000,
        "online_anchor_ba_iters": 100,
        "online_anchor_ba_candidate_multiplier": 2,
        "final_submap_ba_num_pts": 10000,
        "final_submap_ba_iters": 400,
        "final_submap_ba_backend": "pycolmap",
        "final_submap_ba_pycolmap_solver": "dense-schur",
        "final_submap_ba_candidate_multiplier": 3,
        "final_submap_ba_min_track_obs": 3,
        "final_submap_ba_anchor_match_frames": 8,
        "final_submap_ba_max_depth": -1,
        "init_focal": cal["focal"],
        "init_cx": cal["cx"],
        "init_cy": cal["cy"],
        "downsampling": a.downsampling,
        "scene_scale": a.scene_scale,
        "online_pose_export_dir": a.output / "online_colmap",
        "final_submap_ba_export_dir": pose_dir / "final_ba_colmap",
        "pose_debug_ba_point_size": 0.05,
        "pose_debug_gt_point_size": 0.05,
        "pose_debug_frustum_scale": 0.4,
        "pose_debug_ba_frustum_scale": 0.6,
        "pose_debug_gt_frustum_scale": 0.6,
        "profile_timing_dir": pose_dir / "profiling",
    }
    if a.gt_model:
        values["pose_debug_gt_colmap_dir"] = a.gt_model
    for key, value in values.items():
        pose.extend(["--" + key, str(value)])
    flags = [
        "debug_pnp_matching",
        "pose_debug_only",
        "pose_debug_all_frames",
        "keyframe_selection_use_3d",
        "lock_seed_anchor_poses",
        "final_submap_ba",
        "pose_ba_then_reconstruct",
        "fix_focal",
        "profile_timing",
    ]
    flags.append("pose_debug_no_viewer" if a.headless else "pose_debug_viewer")
    if a.exit_after_export or not a.pose_only:
        flags.append("pose_debug_exit_after_export")
    pose.extend("--" + flag for flag in flags)
    stages.append(("04_ground", ROOT / "anchorRecon", pose))
    return stages


def splatting_command(a):
    cmd = [a.splatting_python, '-u', 'train.py',
           '--skyanchor_run', str(a.splatting_only or a.output),
           '-m', str(a.output/'splatting'), '--data_device', 'cpu', '--eval',
           '--dense_init_ply_name', 'dense_v2.ply' if a.splatting_init == 'dense' else 'points3D.ply',
           '--aerial_stage_iterations', str(a.aerial_stage_iterations),
           '--ground_stage_iterations', str(a.ground_stage_iterations),
           '--joint_finetune_iterations', str(a.joint_finetune_iterations),
           '--sky_model', str(a.sky_model)]
    if a.skip_sky_removal:
        cmd.append('--skip_sky_removal')
    if a.splatting_prepare_only:
        cmd.append('--prepare_only')
    return ('05_splatting', ROOT/'splatting', cmd)


def run_stage(stage, logs):
    name, cwd, cmd = stage
    print(f"\n[{name}] cwd={cwd}\n{shlex.join(cmd)}", flush=True)
    with (logs / f"{name}.log").open("wb") as log:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            # Preserve carriage returns so tqdm can refresh the current line.
            while chunk := process.stdout.read(65536):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {logs / (name + '.log')}")


def main(argv=None):
    a = get_args(argv)
    stages = [] if a.splatting_only else commands(a)
    print(f"SkyAnchor output: {a.output}", flush=True)
    if a.dry_run:
        for name, cwd, cmd in stages + ([] if a.pose_only else [splatting_command(a)]):
            print(f"\n[{name}] cwd={cwd}\n{shlex.join(cmd)}")
        return 0
    required = [a.ground_images, a.intrinsics]
    required += (
        [a.ag_model]
        if a.ag_model
        else [
            a.aerial_images,
            a.aerial_model,
            ROOT / "agRecon/configs" / (a.ag_config + ".yaml"),
        ]
    )
    if a.splatting_only:
        required = [a.splatting_only/"run.json", a.splatting_only/"ground/final_ba_colmap"]
    if not a.pose_only and not a.skip_sky_removal:
        required.append(a.sky_model)
    if a.gt_model:
        required.append(a.gt_model)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        raise ValueError(
            f"Output is not empty: {a.output}. Choose a new --output directory."
        )
    logs = a.output / "logs"
    logs.mkdir(parents=True)
    (a.output / "ag").mkdir()
    (a.output / "run.json").write_text(
        json.dumps(vars(a), default=str, indent=2) + "\n"
    )
    executed = []
    if not a.splatting_only:
        for stage in stages[:-1]:
            run_stage(stage, logs)
            executed.append(stage)
        manifest = json.loads((a.output / 'handoff.json').read_text())
        ground = commands(a, manifest['intrinsics'])[-1]
        executed.append(ground)
    if not a.pose_only:
        executed.append(splatting_command(a))
    (a.output/'commands.json').write_text(json.dumps(
        [{'stage': n, 'cwd': str(c), 'argv': cmd} for n,c,cmd in executed], indent=2)+'\n')
    if not a.splatting_only:
        run_stage(ground, logs)
    if not a.pose_only:
        run_stage(splatting_command(a), logs)
    print(f"\nSkyAnchor finished. Results: {a.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Logs and completed exports were preserved.", file=sys.stderr
        )
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"SkyAnchor error: {error}", file=sys.stderr)
        raise SystemExit(1)
