from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from colmap_io import _module as io, read_model
from prepare_inputs import prepare_dataset
from apply_sky_mask import process_one
from selectpairs import select_pairs


def fixture(tmp_path):
    model, images = tmp_path/'model', tmp_path/'images'
    model.mkdir(); images.mkdir()
    cam=io.Camera(4,'PINHOLE',16,12,np.array([20.,20.,7.25,5.25]))
    ims={}
    for iid,tx in [(10,0.),(42,1.),(99,2.)]:
        Image.new('RGB',(32,24),(100,120,150)).save(images/f'{iid}.jpg')
        ims[iid]=io.Image(iid,np.array([1.,0,0,0]),np.array([tx,0,0]),4,f'{iid}.jpg',np.array([[7.,5.]]),np.array([7]))
    pts={7:io.Point3D(7,np.array([0.,0.,5.]),np.array([100,120,150]),0.2,np.array([10,42,99]),np.array([0,0,0]))}
    io.write_model({4:cam},ims,pts,str(model),'.txt')
    return model,images


def test_staging_preserves_ids_tracks_calibration_and_scales(tmp_path):
    model,images=fixture(tmp_path)
    out=prepare_dataset(model,images,tmp_path/'out',scale=2,workers=1)
    c,i,p=read_model(str(out/'sparse/0'))
    assert set(i)=={10,42,99}
    assert i[42].name=='42.png'
    assert Image.open(out/'images/42.png').size==(16,12)
    np.testing.assert_array_equal(c[4].params,[20,20,7.25,5.25])
    np.testing.assert_array_equal(i[42].tvec,[2,0,0])
    np.testing.assert_array_equal(p[7].xyz,[0,0,10])
    np.testing.assert_array_equal(p[7].image_ids,[10,42,99])
    assert (out/'pair.txt').read_text().splitlines()[1::2]==['10','42','99']
    assert not (model/'points3D.ply').exists()
    assert prepare_dataset(model,images,tmp_path/'out',scale=2,workers=1)==out
    with pytest.raises(ValueError,match='Inputs changed'):
        prepare_dataset(model,images,tmp_path/'out',scale=3,workers=1)


def test_no_self_or_zero_score_pairs(tmp_path):
    model,images=fixture(tmp_path)
    c,i,p=read_model(str(model))
    i[99]=i[99]._replace(point3D_ids=np.array([-1]))
    io.write_model(c,i,p,str(model),'.txt')
    path=select_pairs(model,tmp_path/'pair.txt',workers=1)
    lines=path.read_text().splitlines()
    assert lines[-2:]==['99','0 ']
    assert lines[2].split()==['1','42','1']


def test_alpha_mask_convention(tmp_path):
    image=tmp_path/'im.png'; mask=tmp_path/'mask.png'; output=tmp_path/'out.png'
    Image.new('RGB',(2,1),(100,120,140)).save(image)
    Image.fromarray(np.array([[0,255]],dtype=np.uint8)).save(mask)
    process_one(image,mask,output,128)
    a=np.array(Image.open(output))
    np.testing.assert_array_equal(a[0,:,3],[0,255])
    np.testing.assert_array_equal(a[0,:,:3],[[100,120,140],[100,120,140]])


def test_missing_images_fail_before_training(tmp_path):
    model,images=fixture(tmp_path); (images/'42.jpg').unlink()
    with pytest.raises(FileNotFoundError):
        prepare_dataset(model,images,tmp_path/'out',workers=1)


def test_missing_pair_ids_rejected(tmp_path):
    model,images=fixture(tmp_path)
    pair=tmp_path/'bad.txt'; pair.write_text('3\n10\n1 42 10\n42\n1 1 10\n99\n0\n')
    with pytest.raises(ValueError,match='pair neighbors'):
        prepare_dataset(model,images,tmp_path/'out',pair_file=str(pair),workers=1)


def test_root_stage_commands_and_pose_only():
    sys.path.insert(0,str(ROOT.parent))
    from run_skyanchor import get_args,commands,splatting_command
    a=get_args(['--headless','--aerial-stage-iterations','2','--ground-stage-iterations','3','--joint-finetune-iterations','4'])
    cmd=splatting_command(a)[2]
    assert cmd[cmd.index('--aerial_stage_iterations')+1]=='2'
    assert cmd[cmd.index('--ground_stage_iterations')+1]=='3'
    assert cmd[cmd.index('--joint_finetune_iterations')+1]=='4'
    assert '--dense_init_ply_name' in cmd
    assert '--pose_debug_exit_after_export' in commands(a)[-1][2]
    a=get_args(['--pose-only'])
    assert '--pose_debug_exit_after_export' not in commands(a)[-1][2]
    assert '--pose_debug_viewer' in commands(a)[-1][2]


def test_skyremoval_import_does_not_shadow_training_utils():
    import importlib
    import utils
    from submodules.SkyRemoval import skyfilter
    assert skyfilter.utils.__name__=='submodules.SkyRemoval.utils'
    assert importlib.import_module('utils') is utils


def test_odd_image_resize_matches_pose_loader(tmp_path):
    import cv2
    model,images=fixture(tmp_path)
    pixels=np.random.default_rng(0).integers(0,256,(25,32,3),dtype=np.uint8)
    for p in images.iterdir():
        Image.fromarray(pixels).save(p,format='PNG')
    out=prepare_dataset(model,images,tmp_path/'out',workers=1,downsampling=2)
    expected=cv2.resize(pixels,(0,0),fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(np.array(Image.open(out/'images/10.png')),expected)


def test_gpu_projection_preserves_colmap_principal_point(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA camera matrices require GPU')
    from argparse import Namespace
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import loadCam
    model,images=fixture(tmp_path)
    out=prepare_dataset(model,images,tmp_path/'out',workers=1)
    cameras,poses,_=read_model(str(out/'sparse/0'))
    iid=10
    info=readColmapCameras({iid:poses[iid]},cameras,str(out/'images'))[0]
    cam=loadCam(Namespace(resolution=2,data_device='cpu'),0,info,1)
    xyz=-poses[iid].qvec2rotmat().T@poses[iid].tvec+poses[iid].qvec2rotmat().T@np.array([0,0,5])
    clip=torch.tensor(np.r_[xyz,1],dtype=torch.float32,device='cuda')@cam.full_proj_transform
    xy=((clip[:2]/clip[3]+1)*torch.tensor([cam.image_width,cam.image_height],device='cuda')-1)/2
    np.testing.assert_allclose(xy.cpu().numpy(),[(info.cx+.5)/2-.5,(info.cy+.5)/2-.5],atol=1e-5)
