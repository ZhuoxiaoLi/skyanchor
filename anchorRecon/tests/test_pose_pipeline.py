"""Regression coverage for the pose-only entry point, graph indexing and exports."""
from pathlib import Path
from types import SimpleNamespace
import shlex
import subprocess
import sys

import numpy as np
import pytest
import torch

from args import get_args
from dataloaders.read_write_model import read_model
from poses.colmap_export import export_final_ba_colmap_text, export_pose_snapshots_colmap_text
from poses.pipeline import remap_submap_keyframes_to_global
from poses.pycolmap_final_ba import run_final_ba

ROOT = Path(__file__).resolve().parents[1]


def example_args():
    text = (ROOT / 'test.sh').read_text()
    start = text.index('\npython train.py \\\n') + 1
    block = text[start:text.index('\n# python train.py', start)]
    return shlex.split(block.replace('\\\n', ' '), comments=True)[2:]


def test_original_command_compatibility():
    args = get_args(example_args())
    assert args.num_kpts == 4500
    assert args.final_submap_ba_max_depth == -1
    assert args.pose_debug_viewer
    assert args.num_seed_anchor_frames == args.num_tail_anchor_frames == 6
    assert args.final_submap_ba_backend == 'pycolmap'
    args = get_args(example_args() + ['--pose_debug_no_viewer'])
    assert not args.pose_debug_viewer
    assert args.pose_debug_exit_after_export


@pytest.mark.parametrize('options', [
    ['--viewer_mode', 'local'], ['--matcher_backend', 'xfeat'],
    ['--num_iterations', '30'], ['--anchor_front', '9'],
])
def test_removed_or_invalid_modes_fail_early(options):
    with pytest.raises(SystemExit):
        get_args(example_args() + options)


def test_entry_import_does_not_load_gaussian_dependencies():
    script = '''
import sys
import train
import poses.pipeline
assert not any(name in sys.modules for name in (
    'diff_gaussian_rasterization', 'simple_knn', 'lpips', 'fused_ssim',
    'graphdecoviewer', 'cupy', 'scene.mono_depth', 'scene.dense_extractor'))
'''
    subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True)


class Frame:
    def __init__(self, index, translation=0):
        self.index = index
        self.width, self.height = 640, 480
        self.f = torch.tensor([500.])
        self.centre = torch.tensor([319.5, 239.5])
        self.Rt = torch.eye(4)
        self.Rt[0, 3] = translation
        self.info = {'name': f'{index:04d}.png', 'online_Rt': self.Rt.clone(),
                     'online_focal': self.f.clone()}
        self.desc_kpts = SimpleNamespace(matches={})

    def get_Rt(self):
        return self.Rt


def test_submap_remap_and_online_final_exports(tmp_path):
    frames = [Frame(0), Frame(1, 1)]
    frames[0].desc_kpts.matches = {1: 'within submap', 99: 'outside submap'}
    payload = {'keyframes': frames, 'uvs': torch.tensor([[[320., 240.], [420., 240.]]]),
               'points3d': torch.tensor([[0., 0., 5.]]), 'rgb': torch.tensor([[10, 20, 30]])}
    remap_submap_keyframes_to_global(frames, 7)
    assert [f.index for f in frames] == [7, 8]
    assert frames[0].desc_kpts.matches == {8: 'within submap'}
    frames[1].Rt[0, 3] = 2
    export_final_ba_colmap_text(str(tmp_path / 'final'), frames, [payload])
    export_pose_snapshots_colmap_text(str(tmp_path / 'online'), frames)
    cameras, images, points = read_model(str(tmp_path / 'final'), ext='.txt')
    _, online, online_points = read_model(str(tmp_path / 'online'), ext='.txt')
    assert len(cameras) == len(images) == 2
    assert len(points) == 1 and len(online_points) == 0
    np.testing.assert_allclose(images[2].tvec, [2, 0, 0])
    np.testing.assert_allclose(online[2].tvec, [1, 0, 0])
    np.testing.assert_array_equal(points[1].image_ids, [1, 2])
    for image_id, image in images.items():
        assert image.point3D_ids.tolist() == [1]
        assert image_id in points[1].image_ids


def test_pycolmap_ba_preserves_anchors_and_recovers_free_pose():
    torch.manual_seed(3)
    xyz = torch.rand(40, 3)
    xyz[:, :2] -= .5
    xyz[:, 2] += 4
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.tensor([.2, -.5, .5])
    cam_points = xyz[:, None, :] + poses[None, :, :3, 3]
    centre = torch.tensor([319.5, 239.5])
    uvs = 500 * cam_points[:, :, :2] / cam_points[:, :, 2:] + centre
    initial = poses.clone()
    initial[0, 0, 3] += .04
    result = run_final_ba(uvs, xyz + .005, torch.ones(40, dtype=torch.bool), initial,
                          640, 480, 500., centre, n_movable=1)
    assert result['solution_usable']
    assert result['residual'] < 1e-3
    torch.testing.assert_close(result['refined_Rts'][1:], poses[1:])
    torch.testing.assert_close(result['refined_Rts'][0], poses[0], atol=1e-4, rtol=1e-4)
