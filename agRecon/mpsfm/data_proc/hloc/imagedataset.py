import collections.abc as collections
import glob
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import PIL.Image
import torch

from mpsfm.utils.io import read_image
from mpsfm.utils.parsers import parse_image_lists


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += glob.glob((Path(root) / "**" / g).as_posix(), recursive=True)
            if len(paths) == 0:
                raise ValueError(f"Could not find any image in root: {root}.")
            paths = sorted(set(paths))
            self.names = [Path(p).relative_to(root).as_posix() for p in paths]
            print("Found {len(self.names)} images in root {root}.")
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p for p in paths]
            else:
                raise ValueError(f"Unknown format for path argument {paths}.")

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(f"Image {name} does not exists in root: {root}.")

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        image = image[None] if self.conf.grayscale else image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)

class AGImageDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }

    def __init__(self, imnames2path, conf, paths):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.imnames2path = imnames2path
        self.names = paths

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(Path(self.imnames2path[name]), self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        image = image[None] if self.conf.grayscale else image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)

import torchvision.transforms as T
from PIL import Image
class AGSImageDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }

    def __init__(self, imnames2path, conf, paths):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.imnames2path = imnames2path
        self.names = paths

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(Path(self.imnames2path[name]), self.conf.grayscale)
        size = image.shape[:2][::-1]  # (W, H)

        if self.conf.resize_max and (self.conf.resize_force or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            raise ValueError("cannot use grayscale images with RGB mean/std normalize.")
            # image = np.repeat(image[:, :, None], 3, axis=2)

        def crop_to_multiple_np(img_hwc, m=14):
            H, W = img_hwc.shape[:2]
            H2 = (H // m) * m
            W2 = (W // m) * m

            if H2 == 0 or W2 == 0:
                raise ValueError(
                    f"Image too small after resize for patch size {m}: got (H,W)=({H},{W}). "
                    f"Need at least {m}x{m}."
                )

            dh = H - H2
            dw = W - W2
            return img_hwc[:H2, :W2, :], (dh, dw)  # (crop_bottom, crop_right)

        image, (crop_bottom, crop_right) = crop_to_multiple_np(image, m=14)
        cropped_size = image.shape[:2][::-1]  # (W_crop, H_crop)

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        pil_img = Image.fromarray(image, mode="RGB")
        image_t = self.input_transform()(pil_img)

        data = {
            "image": image_t,
            "original_size": np.array(size),
            "cropped_size": np.array(cropped_size),
            "crop_bottom_right": np.array([crop_bottom, crop_right])
        }
        return data

    def __len__(self):
        return len(self.names)

    def input_transform(self):
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
