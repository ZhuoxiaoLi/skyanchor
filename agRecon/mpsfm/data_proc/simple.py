"""Dataset and parser for reconstructing user provided images."""

from pathlib import Path

import pycolmap
import torch
import yaml
from PIL import Image
from pycolmap import Reconstruction

from .basedataset import BaseDataset, BaseDatasetParser


class SimpleDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for user provided images following standard pipeline structure."""

    def _init(self, image_list=None, scene_parser=None, images_dir=None, **kwargs):
        assert self.conf.batch_size == 1, "Batch size must be 1 for this dataset"
        if scene_parser is None:
            assert images_dir is not None, "images_dir must be provided if scene_parser is None"
            self.scene_parser = SimpleParser(rgb_dir=images_dir.parent, imnames=image_list)
        else:
            self.scene_parser = scene_parser
        if image_list is None:
            image_ids = list(self.scene_parser.rec.images.keys())
        else:
            image_ids = [k for k, v in self.scene_parser.rec.images.items() if v.name in image_list]

        image_ids.sort()
        self.imids = image_ids


class SimpleParser(BaseDatasetParser):
    """Parser for user provided images following standard pipeline structure."""

    scene = "<custom>"
    dataset = SimpleDataset

    def __init__(self, *args, data_dir=None, imnames=None, intrinsics_pth=None, rgb_dir=None, **kwargs):
        self.rec = Reconstruction()
        self.reconstruction_dir = None
        if rgb_dir is None:
            rgb_dir = Path(data_dir) / "images"
        else:
            rgb_dir = Path(rgb_dir)
        if imnames is None:
            imnames = []
            imname2path = {}
            for p in rgb_dir.rglob('*'):
                if p.is_file():
                    name = str(p.relative_to(rgb_dir))
                    imnames.append(name)
                    imname2path[name] = p

        self.imnames = imnames
        self.imname2path = imname2path

        if intrinsics_pth is None:
            intrinsics_pth = Path(data_dir) / "intrinsics.yaml"
        else:
            intrinsics_pth = Path(intrinsics_pth)
        with open(intrinsics_pth, encoding="utf-8") as f:
            intrinsics = yaml.safe_load(f)
        if len(intrinsics) == 1:
            intr_id, intr = next(iter(intrinsics.items()))
            assert intr["images"] == "all" or (
                isinstance(intr["images"], list) and len(intr["images"]) == len(self.imnames)
            ), "If only one camera is provided, images must be 'all' or a list of all image names"

        image_id = 1
        for camid, camdict in intrinsics.items():
            params = camdict["params"]
            images = camdict["images"]
            if images == "all":
                images = self.imnames

            shapes = []
            for imname in images:
                im_path = rgb_dir / imname
                if not im_path.exists():
                    continue

                im = Image.open(im_path)
                shapes.append(im.size)
            assert len(set(shapes)) == 1, "All images must have the same shape"
            width, height = shapes[0]

            camera = pycolmap.Camera(
                model="PINHOLE",
                width=width,
                height=height,
                params=params,
                has_prior_focal_length=True,
                camera_id=camid,
            )
            self.rec.add_camera(camera)

            for imanme in images:
                self.rec.add_image(pycolmap.Image(name=imanme, camera_id=camid, image_id=image_id))
                image_id += 1

        self.rgb_dir = rgb_dir
