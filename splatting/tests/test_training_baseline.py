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
    for s2, s3, expected in [(False, False, (2000, 20000)), (True, False, (32000, 50000)), (True, True, (100000, 100000))]:
        result = compute_densify_window_and_params(opt, has_ground=True, ground_start_iter=30000, total_iterations=100000, stage2_active=s2, stage3_active=s3)
        assert result[:2] == expected
        assert result[2:4] == ((0, 0) if s3 else (600, 9000))
    assert opt.budget == 8000000


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
    model.reset_opacity()
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


@pytest.mark.parametrize('optimizer', ['default', 'sparse_adam'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optimizers required')
def test_independent_ground_stage_and_joint_merge(optimizer):
    opt = options()
    opt.optimizer_type = optimizer
    xyz = np.random.default_rng(12).normal(size=(8, 3)).astype(np.float32)
    rgb = np.full_like(xyz, .5)
    pcd = BasicPointCloud(xyz, rgb, np.zeros_like(xyz))
    aerial = GaussianModel(1, optimizer)
    aerial.create_from_pcd(pcd, 2.0)
    aerial.training_setup(opt)
    aerial.oneupSHdegree()

    def step(model):
        for name in model.parameter_names:
            getattr(model, name).grad = torch.ones_like(getattr(model, name))
        model.optimizer_step(torch.ones(len(model.get_xyz), dtype=torch.bool, device='cuda'))

    step(aerial)
    cached = aerial.cache_parameters()
    expected_aerial = {name: value.clone() for name, value in cached['parameters'].items()}
    assert all(value.device.type == 'cpu' and not value.requires_grad
               for value in cached['parameters'].values())
    ground = GaussianModel(1, optimizer)
    ground.create_from_pcd(pcd, 3.0)
    ground.training_setup(opt)
    assert len(ground.get_xyz) == len(xyz)
    assert ground.active_sh_degree == 0
    assert not ground.optimizer.state
    assert ground.spatial_lr_scale == 3.0
    step(ground)
    ground.densification_postfix(*ground.point_parameters(xyz + 10, rgb))
    ground.reset_opacity()
    step(ground)
    for name in ground.parameter_names:
        torch.testing.assert_close(cached['parameters'][name], expected_aerial[name], rtol=0, atol=0)
    expected_ground = {name: getattr(ground, name).detach().clone() for name in ground.parameter_names}
    ground.merge_cached_parameters(cached, opt)
    assert len(ground.get_xyz) == 3 * len(xyz)
    assert ground.active_sh_degree == 1
    assert ground._n_frozen == 0
    assert not ground.optimizer.state
    for name in ('_xyz', '_features_dc', '_features_rest', '_rotation'):
        torch.testing.assert_close(getattr(ground, name)[:len(xyz)].cpu(), expected_aerial[name], rtol=0, atol=0)
        torch.testing.assert_close(getattr(ground, name)[len(xyz):], expected_ground[name], rtol=0, atol=0)
    distances = torch.cdist(ground.get_xyz, ground.get_xyz).sort(dim=1).values[:, 1:4]
    expected_scale = distances.square().mean(1).clamp_min(1e-7).sqrt()
    torch.testing.assert_close(ground.get_scaling, expected_scale[:, None].repeat(1, 3))
    torch.testing.assert_close(ground.get_opacity, torch.full_like(ground.get_opacity, .1))
    merged = {name: getattr(ground, name).detach().clone() for name in ground.parameter_names}
    step(ground)
    for name in ground.parameter_names:
        assert not torch.equal(getattr(ground, name)[:len(xyz)], merged[name][:len(xyz)])
        assert not torch.equal(getattr(ground, name)[len(xyz):], merged[name][len(xyz):])


@pytest.mark.parametrize('optimizer', ['default', 'sparse_adam'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optimizers required')
def test_opacity_cap_and_split_prune_accounting(optimizer):
    opt = options()
    assert opt.opacity_reset_max == .01
    assert opt.cull_opacity == .005
    model = GaussianModel(1, optimizer)
    xyz = np.random.default_rng(21).normal(size=(8, 3)).astype(np.float32)
    model.create_from_pcd(BasicPointCloud(xyz, np.full_like(xyz, .5), np.zeros_like(xyz)), 1.0)
    model.training_setup(opt)
    original = torch.tensor([.8, .002, .003, .006, .007, .008, .9, .8], device='cuda')[:, None]
    with torch.no_grad():
        model._opacity.copy_(model.inverse_opacity_activation(original))
    model.reset_opacity()
    torch.testing.assert_close(model.get_opacity, original.clamp_max(.01))
    model.xyz_gradient_accum.fill_(1)
    model.denom.fill_(1)
    scores = torch.zeros(8, device='cuda')
    scores[0] = 1
    result = model.densify_and_prune_Improved(scores, .005, 9, opt, 100)
    assert result == {'split': 1, 'pruned': 2}
    assert len(model.get_xyz) == 8 + result['split'] - result['pruned']
    assert (model.get_opacity >= .005).all()
    result = model.densify_and_prune_Improved(torch.zeros(7, device='cuda'), .005, 7, opt, 200)
    assert result == {'split': 0, 'pruned': 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rasterization required')
def test_opacity_reset_keeps_translucent_gaussians_trainable():
    from scene.cameras import Camera
    from gaussian_renderer import render
    opt = options()
    xyz = np.array([[x, y, 2.] for x in [-.25, .25] for y in [-.25, .25]], dtype=np.float32)
    model = GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(xyz, np.full_like(xyz, .5), np.zeros_like(xyz)), 1.)
    model.training_setup(opt)
    with torch.no_grad():
        model._scaling.fill_(np.log(.2))
        model._opacity.copy_(model.inverse_opacity_activation(
            torch.tensor([.1, .2, .3, .9], device='cuda')[:, None]))
    model.reset_opacity(opt.opacity_reset_max)
    camera = Camera(0, np.eye(3), np.zeros(3), np.pi/2, np.pi/2,
                    torch.full((3, 64, 64), .2), torch.ones(1, 64, 64),
                    'opacity_reset', 0, data_device='cpu')
    pipe = Namespace(separate_sh=True, convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    pkg = render(camera, model, pipe, torch.zeros(3, device='cuda'))
    keep = pkg['radii'] > 0
    assert keep.all()
    (pkg['render'] - .2).square().mean().backward()
    assert (model._opacity.grad < 0).all()
    previous_opacity = model.get_opacity.detach().clone()
    model.optimizer_step(keep)
    assert (model.get_opacity > previous_opacity).all()
    with torch.no_grad():
        result = model.densify_and_prune_Improved(torch.ones(4, device='cuda'), .005, 4, opt, 300)
    assert result == {'split': 0, 'pruned': 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rasterization required')
def test_offscreen_gaussian_keeps_masked_pixel_color_gradients():
    from scene.cameras import Camera
    from gaussian_renderer import render
    from train import photometric_loss
    alpha = torch.zeros(1, 64, 64)
    alpha[:, 16:48, 16:48] = 1
    camera = Camera(0, np.eye(3), np.zeros(3), np.pi/2, np.pi/2,
                    torch.zeros(3, 64, 64), alpha, 'test', 0, data_device='cpu')
    model = GaussianModel(3)
    xyz = np.array([[3, 0, 2], [3.1, 0, 2], [3, .1, 2], [3, 0, 2.1]], dtype=np.float32)
    model.create_from_pcd(BasicPointCloud(xyz, np.ones_like(xyz), np.zeros_like(xyz)), 1.0)
    with torch.no_grad():
        model._scaling.fill_(0)
    model.training_setup(options())
    pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, separate_sh=True)
    background = torch.zeros(3, device='cuda')
    pkg = render(camera, model, pipe, background)
    assert (pkg['gaussian_centers'][:, 0] >= 64).all()
    keep = pkg['radii'] > 0
    assert keep.any(), 'Offscreen centers can still cover supervised pixels'
    loss = (pkg['render'] * alpha.cuda()).mean()
    before = loss.detach().clone()
    loss.backward()
    expected = model._features_dc.grad.clone()
    assert expected.abs().sum() > 0
    model.optimizer_step(keep)
    with torch.no_grad():
        after = (render(camera, model, pipe, background)['render'] * alpha.cuda()).mean()
    assert after < before
    camera.alpha_mask.zero_()
    pkg = render(camera, model, pipe, background)
    loss, _, _, _ = photometric_loss(pkg['render'], camera, .2)
    loss.backward()
    for name in model.parameter_names:
        assert torch.count_nonzero(getattr(model, name).grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA camera and SSIM required')
def test_soft_alpha_is_applied_once_to_rgb_and_loss():
    from scene.cameras import Camera
    from train import photometric_loss
    rgb = torch.full((3, 16, 16), .6)
    alpha = torch.full((1, 16, 16), .5)
    alpha[:, :, :8] = 0
    camera = Camera(0, np.eye(3), np.zeros(3), np.pi/2, np.pi/2,
                    rgb, alpha, 'soft_alpha', 0, data_device='cpu')
    torch.testing.assert_close(camera.original_image, rgb)
    prediction = rgb.cuda().clone().requires_grad_()
    loss, l1, masked_prediction, masked_target = photometric_loss(prediction, camera, .2)
    assert l1 == 0
    torch.testing.assert_close(masked_prediction, masked_target)
    with torch.no_grad():
        prediction.add_(.1)
    loss, _, _, _ = photometric_loss(prediction, camera, .2)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :8]) == 0
    assert prediction.grad[:, :, 8:].abs().sum() > 0


def test_densification_sampling_covers_the_sequence_and_keeps_edges_paired():
    import random
    from train import sample_densification_views
    state = random.getstate()
    try:
        random.seed(42)
        cameras = list(range(100))
        edges = [i + 1000 for i in cameras]
        remaining = []
        selected, selected_edges = sample_densification_views(cameras, edges, remaining, 6)
        assert min(selected) < 50 and max(selected) >= 50
        assert selected_edges == [i + 1000 for i in selected]
        rest, rest_edges = sample_densification_views(cameras, edges, remaining, 94)
        assert sorted(selected + rest) == cameras
        assert rest_edges == [i + 1000 for i in rest]
        assert not remaining
        again, again_edges = sample_densification_views(cameras, edges, remaining, 100)
        assert sorted(again) == cameras
        assert again_edges == [i + 1000 for i in again]
    finally:
        random.setstate(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA training required')
def test_aerial_and_ground_use_identical_training_updates(monkeypatch):
    import train
    from scene.cameras import Camera
    xyz = np.array([[center + x, y, 2.] for center in [0., 5.]
                    for x in [-.1, .1] for y in [-.1, .1]], dtype=np.float32)
    pcd = BasicPointCloud(xyz, np.full_like(xyz, .5), np.zeros_like(xyz))
    create_from_pcd = GaussianModel.create_from_pcd
    def initialize(model, *args, **kwargs):
        create_from_pcd(model, *args, **kwargs)
        # Avoid the undefined orientation of isotropic Gaussians in the parity check.
        with torch.no_grad():
            model._scaling.add_(torch.tensor([0., .2, -.3], device='cuda'))
    monkeypatch.setattr(GaussianModel, 'create_from_pcd', initialize)
    target = torch.from_numpy(np.random.default_rng(8).uniform(.1, .4, (3, 32, 32)).astype(np.float32))
    def cameras(with_alpha):
        return [Camera(i, np.eye(3), np.array([-center, 0., 0.]), np.pi/3, np.pi/3,
                       target, torch.ones(1, 32, 32) if with_alpha else None,
                       str(i), i, data_device='cpu') for i, center in enumerate([0., 5.])]
    aerial_cameras, ground_cameras = cameras(False), cameras(True)
    saved = {}
    class Scene:
        def __init__(self, dataset, gaussians):
            self.gaussians = gaussians
            self.cameras_extent = 1.
            gaussians.create_from_pcd(pcd, 1., dataset.init_scale_factor, dataset.init_opacity)
        def getTrainCameras(self):
            return aerial_cameras
        def getTestCameras(self):
            return []
        def save(self, iteration):
            model = self.gaussians
            saved[iteration] = {name: getattr(model, name).detach().clone() for name in model.parameter_names}
            saved[iteration]['gradient_accum'] = model.xyz_gradient_accum.clone()
            saved[iteration]['denom'] = model.denom.clone()
            saved[iteration]['moments'] = {group['name']: {
                key: value.clone() for key, value in optimizer.state[group['params'][0]].items()
            } for optimizer in [model.optimizer, model.shoptimizer] for group in optimizer.param_groups}
    monkeypatch.setattr(train, 'Scene', Scene)
    monkeypatch.setattr(train, 'prepare_output_and_logger', lambda *_: None)
    monkeypatch.setattr(train, 'maybe_run_dense_initialization', lambda *args: None)
    monkeypatch.setattr(train, 'ensure_ground_dense_init_ply', lambda *args: None)
    monkeypatch.setattr(train, 'load_ground_cameras', lambda *args: (ground_cameras, [], 1.))
    monkeypatch.setattr(train, 'load_ground_point_cloud', lambda *args: pcd)
    monkeypatch.setattr(train, 'training_report', lambda *args, **kwargs: None)
    monkeypatch.setattr(train, 'randint', lambda a, b: a)
    opt = options()
    opt.aerial_stage_iterations = opt.ground_stage_iterations = 4
    opt.joint_finetune_iterations = 2
    opt.densify_from_iter, opt.densify_until_iter = 0, 4
    opt.densification_interval, opt.opacity_reset_interval = 2, 3
    opt.budget = len(xyz)
    dataset = Namespace(sh_degree=3, ground_source_path='ground', ground_images='images',
                        white_background=False, init_scale_factor=1., init_opacity=.1)
    pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, separate_sh=True)
    train.training(dataset, opt, pipe, [], [4, 8, 9, 10], -1, False, Namespace(cams=1))
    for name in GaussianModel.parameter_names + ('gradient_accum', 'denom'):
        torch.testing.assert_close(saved[4][name], saved[8][name], rtol=1e-5, atol=1e-7)
    for name in saved[4]['moments']:
        for key in saved[4]['moments'][name]:
            torch.testing.assert_close(saved[4]['moments'][name][key], saved[8]['moments'][name][key], rtol=1e-5, atol=1e-7)
    assert len(saved[9]['_xyz']) == len(saved[10]['_xyz']) == 2 * len(xyz)
