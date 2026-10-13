#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = tf.to_tensor(Image.open(renders_dir / fname)).unsqueeze(0).cuda()
        gt = tf.to_tensor(Image.open(gt_dir / fname)).unsqueeze(0).cuda()
        render_rgb = render[:, :3, :, :]
        gt_rgb = gt[:, :3, :, :]
        alpha = gt[:, 3:4, :, :] if gt.shape[1] >= 4 else None
        if alpha is None and render.shape[1] >= 4:
            alpha = render[:, 3:4, :, :]
        if alpha is not None:
            render_rgb = render_rgb * alpha
            gt_rgb = gt_rgb * alpha
        renders.append(render_rgb)
        gts.append(gt_rgb)
        image_names.append(fname)
    return renders, gts, image_names

def _evaluate_method_dir(method_dir: Path):
    """Compute mean metrics for one {test_xxx}/{method} directory. Return (summary, per_view), or None if inputs are missing."""
    gt_dir = method_dir / "gt"
    renders_dir = method_dir / "renders"
    if not gt_dir.is_dir() or not renders_dir.is_dir():
        print(f"  [skip] missing gt/renders in {method_dir}")
        return None

    renders, gts, image_names = readImages(renders_dir, gt_dir)
    if not renders:
        print(f"  [skip] no images in {method_dir}")
        return None

    ssims, psnrs, lpipss = [], [], []
    for idx in tqdm(range(len(renders)), desc=f"  Eval {method_dir.parent.name}/{method_dir.name}"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    ssim_mean = torch.tensor(ssims).mean().item()
    psnr_mean = torch.tensor(psnrs).mean().item()
    lpips_mean = torch.tensor(lpipss).mean().item()
    print(f"  N={len(renders):<5}  SSIM: {ssim_mean:.5f}   PSNR: {psnr_mean:.5f}   LPIPS: {lpips_mean:.5f}")

    summary = {"N": len(renders), "SSIM": ssim_mean, "PSNR": psnr_mean, "LPIPS": lpips_mean}
    per_view = {
        "SSIM":  {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
        "PSNR":  {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        "LPIPS": {name: l for l, name in zip(torch.tensor(lpipss).tolist(), image_names)},
    }
    return summary, per_view


def evaluate(model_paths):
    print("")

    CANDIDATE_SPLITS = ["test", "test_aerial", "test_ground"]

    for scene_dir in model_paths:
        scene_dir_p = Path(scene_dir)
        print(f"Scene: {scene_dir}")

        full_dict = {}
        per_view_dict = {}

        for split_name in CANDIDATE_SPLITS:
            split_dir = scene_dir_p / split_name
            if not split_dir.is_dir():
                continue
            print(f"  [Split] {split_name}")
            full_dict[split_name] = {}
            per_view_dict[split_name] = {}
            for method in sorted(os.listdir(split_dir)):
                method_dir = split_dir / method
                if not method_dir.is_dir():
                    continue
                result = _evaluate_method_dir(method_dir)
                if result is None:
                    continue
                summary, per_view = result
                full_dict[split_name][method] = summary
                per_view_dict[split_name][method] = per_view

        if not full_dict:
            print(f"  No evaluable splits found in {scene_dir}")
            continue

        print(f"\n  === Summary for {scene_dir} ===")
        print(f"  {'Split':<14} {'Method':<20} {'N':>5}  {'SSIM':>9}  {'PSNR':>9}  {'LPIPS':>9}")
        for split_name, methods in full_dict.items():
            for method, s in methods.items():
                print(f"  {split_name:<14} {method:<20} {s['N']:>5}  "
                      f"{s['SSIM']:>9.5f}  {s['PSNR']:>9.5f}  {s['LPIPS']:>9.5f}")
        print("")

        with open(scene_dir_p / "results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(scene_dir_p / "per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
