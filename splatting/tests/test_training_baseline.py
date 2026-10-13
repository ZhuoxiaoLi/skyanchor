from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from train import apply_explicit_aerial_ground_length, compute_densify_window_and_params


def options():
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args([]))


def test_three_stage_windows():
    opt = options()
    apply_explicit_aerial_ground_length(opt, Namespace(ground_source_path='ground'))
    assert (opt.ground_start_iter, opt.iterations, opt.joint_finetune_iterations) == (30000, 60000, 40000)
    for s2, s3, expected in [(False, False, (2000, 20000)), (True, False, (32000, 50000)), (True, True, (65000, 85000))]:
        result = compute_densify_window_and_params(opt, has_ground=True, ground_start_iter=30000, total_iterations=100000, joint_finetune_iters=40000, stage2_active=s2, stage3_active=s3)
        assert result[:2] == expected
        assert result[2:4] == (300, 6000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA initialization required')
def test_local_spacing_and_cached_ply_agree(tmp_path):
    from scene.dataset_readers import storePly, fetchPly
    xyz = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [2,2,2]], dtype=np.float32)
    rgb = np.full_like(xyz, 128/255)
    model = GaussianModel(1)
    fresh = model.point_parameters(xyz, rgb)
    shifted = model.point_parameters(xyz + np.array([50,-30,100], dtype=np.float32), rgb)
    torch.testing.assert_close(fresh[4], shifted[4])
    expected = torch.cdist(torch.tensor(xyz), torch.tensor(xyz)).sort(dim=1).values[:,1:4].square().mean(1).sqrt()
    torch.testing.assert_close(fresh[4].exp()[:,0].cpu(), expected)
    path = tmp_path/'dense.ply'
    storePly(str(path), xyz, np.full_like(xyz,128))
    cached = GaussianModel(1)
    cached.create_from_pcd(fetchPly(str(path)), 1.0)
    for actual,wanted in zip((cached._xyz,cached._features_dc,cached._features_rest,cached._opacity,cached._scaling,cached._rotation),fresh):
        torch.testing.assert_close(actual,wanted)
    torch.testing.assert_close(cached.get_opacity,torch.full((5,1),.1,device='cuda'))
    half=model.point_parameters(xyz,rgb,.5)
    torch.testing.assert_close(half[4].exp(),fresh[4].exp()*.5)


@pytest.mark.parametrize('optimizer',['default','sparse_adam'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optimizers required')
def test_freeze_survives_momentum_and_tensor_replacement(optimizer):
    opt=options();opt.optimizer_type=optimizer
    model=GaussianModel(1,optimizer)
    xyz=np.random.default_rng(4).normal(size=(8,3)).astype(np.float32)
    rgb=np.full_like(xyz,.5)
    model.create_from_pcd(BasicPointCloud(xyz,rgb,np.zeros_like(xyz)),1.0)
    model.training_setup(opt)
    names=('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation')
    def step():
        for name in names:
            getattr(model,name).grad=torch.ones_like(getattr(model,name))
        model.optimizer_step(torch.ones(len(model._xyz),dtype=torch.bool,device='cuda'))
    step()
    frozen=[getattr(model,n)[:2].detach().clone() for n in names]
    model.freeze_by_mask(2)
    model.densification_postfix(*model.point_parameters(xyz+3,rgb))
    mask=torch.zeros(len(model._xyz),dtype=torch.bool,device='cuda');mask[-1]=True
    model.prune_points(mask)
    model.reset_opacity(.005)
    step()
    for name,expected in zip(names,frozen):
        torch.testing.assert_close(getattr(model,name)[:2],expected,rtol=0,atol=0)
    model.unfreeze_all();step()
    assert not torch.equal(model._xyz[:2],frozen[0])


def test_position_schedule_restarts_per_stage():
    model=GaussianModel(1);model.spatial_lr_scale=2;opt=options()
    for length in (30000,40000):
        model.configure_position_schedule(opt,length)
        assert model.xyz_scheduler_args(0)==pytest.approx(opt.position_lr_init*2)
        assert model.xyz_scheduler_args(length)==pytest.approx(opt.position_lr_final*2)
