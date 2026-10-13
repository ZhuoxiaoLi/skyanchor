import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "anchorRecon"))
from run_skyanchor import commands, get_args, run_stage

spec = importlib.util.spec_from_file_location(
    "export_ag", ROOT / "tools/export_ag_anchors.py"
)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


@pytest.mark.parametrize(
    "count,expected",
    [(263, [50, 50, 50, 50, 63]), (275, [50, 50, 50, 50, 50, 25]), (100, [50, 50])],
)
def test_ag_boundary_rules(count, expected):
    assert [b - a for a, b in bridge.submap_ranges(count, 50, 6, 6)] == expected


def fixture_model(tmp_path):
    from dataloaders.read_write_model import Camera, BaseImage, write_model

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(36):
        Image.new("RGB", (40, 30)).save(image_dir / f"{index:04d}.png")
    yaml = tmp_path / "intrinsics.yaml"
    yaml.write_text("1:\n  params: [50, 50, 20, 15]\n  images: all\n")
    names = bridge.image_names(image_dir)
    wanted = bridge.expected_anchors(names, bridge.submap_ranges(36, 18, 2, 2), 2, 2)
    images = {
        i: BaseImage(
            i,
            np.array([1.0, 0, 0, 0]),
            np.array([float(i), 0, 0]),
            1,
            name,
            np.empty((0, 2)),
            np.empty(0, dtype=np.int64),
        )
        for i, name in enumerate(wanted, 1)
    }
    camera = Camera(1, "PINHOLE", 40, 30, np.array([50.0, 50, 20, 15]))
    model = tmp_path / "model"
    model.mkdir()
    write_model({1: camera}, images, {}, str(model), ext=".bin")
    return image_dir, yaml, model, wanted, images, camera


def test_binary_handoff_exact_poses(tmp_path):
    image_dir, yaml, model, wanted, images, camera = fixture_model(tmp_path)
    data = bridge.export(
        model,
        image_dir,
        yaml,
        tmp_path / "anchors.txt",
        tmp_path / "handoff.json",
        18,
        2,
        2,
        2,
    )
    rows = (tmp_path / "anchors.txt").read_text().splitlines()[1:]
    assert [r.split()[0] for r in rows] == wanted
    for i, row in enumerate(rows, 1):
        np.testing.assert_array_equal(
            np.array(row.split()[1:], float), [*images[i].tvec, *images[i].qvec]
        )
    assert data["submap_ranges"] == [(0, 18), (18, 36)]
    assert data["intrinsics"]["focal"] == 25
    assert data["intrinsics"]["cx"] == 9.75


def test_missing_anchor_blocks_export(tmp_path):
    from dataloaders.read_write_model import write_model

    image_dir, yaml, model, wanted, images, camera = fixture_model(tmp_path)
    del images[1]
    write_model({1: camera}, images, {}, str(model), ext=".bin")
    with pytest.raises(ValueError, match="missing 1 required anchors"):
        bridge.export(
            model,
            image_dir,
            yaml,
            tmp_path / "anchors.txt",
            tmp_path / "handoff.json",
            18,
            2,
            2,
            2,
        )
    assert not (tmp_path / "anchors.txt").exists()


def test_nonsquare_focal_rejected(tmp_path):
    image_dir, yaml, *_ = fixture_model(tmp_path)
    yaml.write_text("1:\n  params: [50, 51, 20, 15]\n  images: all\n")
    with pytest.raises(ValueError, match="single focal"):
        bridge.calibrated_intrinsics(yaml, bridge.image_names(image_dir), image_dir, 2)


def test_generated_ground_cli_and_reuse(tmp_path):
    from args import get_args as get_pose_args

    a = get_args(
        [
            "--ag-model",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "out"),
            "--headless",
        ]
    )
    stages = commands(a, {"focal": 506.03, "cx": 377.25, "cy": 246.5})
    assert [s[0] for s in stages] == ["01_validate", "03_export", "04_ground"]
    pose = get_pose_args(stages[-1][2][3:])
    assert pose.anchor_submap_size == 0
    assert pose.pose_debug_no_viewer and not pose.pose_debug_viewer
    assert pose.init_focal == 506.03
    assert not a.output.exists()


def test_failed_stage_preserves_log_and_fails(tmp_path):
    with pytest.raises(RuntimeError, match="exit 7"):
        run_stage(
            (
                "test",
                tmp_path,
                [sys.executable, "-c", 'print("failed-stage"); raise SystemExit(7)'],
            ),
            tmp_path,
        )
    assert "failed-stage" in (tmp_path / "test.log").read_text()


def test_ag_failure_stops_pipeline(tmp_path, monkeypatch):
    import run_skyanchor as runner

    image_dir, yaml, model, *_ = fixture_model(tmp_path)
    reached = []

    def fail_ag(stage, logs):
        reached.append(stage[0])
        if stage[0] == "02_ag":
            raise RuntimeError("AG failed")

    monkeypatch.setattr(runner, "run_stage", fail_ag)
    with pytest.raises(RuntimeError, match="AG failed"):
        runner.main(
            [
                "--ground-images",
                str(image_dir),
                "--intrinsics",
                str(yaml),
                "--aerial-images",
                str(image_dir),
                "--aerial-model",
                str(model),
                "--output",
                str(tmp_path / "run"),
            ]
        )
    assert reached == ["01_validate", "02_ag"]


def test_launcher_from_other_directory_dry_run(tmp_path):
    import subprocess

    out = tmp_path / "not_created"
    result = subprocess.run(
        [str(ROOT / "run_skyanchor.sh"), "--dry-run", "--output", str(out)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "[02_ag]" in result.stdout and "[04_ground]" in result.stdout
    assert not out.exists()


def test_ag_parser_ignores_download_temporary_files(tmp_path):
    from dataloaders.read_write_model import Camera, BaseImage, write_model

    sys.path.insert(0, str(ROOT / "agRecon"))
    from mpsfm.data_proc.AGData import AGParser

    ground = tmp_path / "ground"
    aerial = tmp_path / "aerial"
    model = tmp_path / "aerial_model"
    for directory in (ground, aerial, model):
        directory.mkdir()
    Image.new("RGB", (40, 30)).save(aerial / "air.png")
    (aerial / "air.png.baiduyun.p.downloading").write_bytes(b"not an image")
    Image.new("RGB", (40, 30)).save(aerial / "unregistered.png")
    for i in range(20):
        Image.new("RGB", (40, 30)).save(ground / f"g{i:03d}.png")
    (ground / "g000.png.baiduyun.p.downloading").write_bytes(b"not an image")
    image = BaseImage(
        1,
        np.array([1.0, 0, 0, 0]),
        np.zeros(3),
        1,
        "air.png",
        np.empty((0, 2)),
        np.empty(0, dtype=np.int64),
    )
    write_model(
        {1: Camera(1, "PINHOLE", 40, 30, np.array([50.0, 50, 20, 15]))},
        {1: image},
        {},
        str(model),
        ext=".bin",
    )
    yaml = tmp_path / "intrinsics.yaml"
    yaml.write_text("1:\n  params: [50, 50, 20, 15]\n  images: all\n")
    parser = AGParser(
        intrinsics_pth=yaml,
        recon_exist_dir=model,
        rgb_register_dir=aerial,
        rgb_unregister_dir=ground,
        submap_im_num=50,
        anchor_num_pair=(6, 6),
    )
    assert parser.imnames_register == ["air.png"]
    assert len(parser.imnames_unregister) == 12
    (aerial / "air.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing 1 aerial reference images"):
        AGParser(
            intrinsics_pth=yaml,
            recon_exist_dir=model,
            rgb_register_dir=aerial,
            rgb_unregister_dir=ground,
        )


def test_geometry_imports_with_unrelated_projects_module():
    import subprocess

    script = """
import sys, types
sys.modules['projects'] = types.ModuleType('projects')
from mpsfm.extraction.imagewise.geometry.models.normals.dsine import get_default_parser
assert get_default_parser.__module__ == 'third_party.DSINE.projects'
from mpsfm.extraction.imagewise.geometry.models.depth.metric3dv2 import Metric3Dv2
assert Metric3Dv2.name == 'metric3dv2'
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT / "agRecon", check=True)
