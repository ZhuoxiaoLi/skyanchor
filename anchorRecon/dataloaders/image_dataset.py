#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging
from argparse import Namespace

from dataloaders.read_write_model import read_model, qvec2rotmat
from utils import get_image_names


class ImageDataset:
    """
    The main dataset class for loading images from disk in a multithreaded manner.
    It also supports loading masks and COLMAP poses if available.
    The next image can be fetched using the `getnext` method.
    """
    def __init__(self, args: Namespace):
        self.images_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.images_dir)
        self.image_name_list.sort()
        self.image_name_list = self.image_name_list[args.start_at :]
        self.image_paths = [
            os.path.join(self.images_dir, image_name)
            for image_name in self.image_name_list
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.mask_dir = (
            os.path.join(args.source_path, args.masks_dir) if args.masks_dir else None
        )
        if self.mask_dir:
            self.mask_paths = [
                os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
                for image_name in self.image_name_list
            ]
            assert all(os.path.exists(mask_path) for mask_path in self.mask_paths), (
                "Not all masks exist."
            )

        self.downsampling = args.downsampling
        self.num_threads = min(args.num_loader_threads, len(self.image_paths))
        self.current_index = 0
        self.preload_queue = Queue(maxsize=self.num_threads)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.fixed_image_size = None
        self.anchor_submap_ranges = []

        self.infos = {
            name: {
                "is_test": (args.test_hold > 0) and (i % args.test_hold == 0),
                "name": name,
                "dataset_index": i,
            }
            for i, name in enumerate(self.image_name_list)
        }

        first_image = self._load_image(self.image_paths[0])
        self.width, self.height = first_image.shape[2], first_image.shape[1]
        res = self.width * self.height
        max_res = 1_500_000  # 1.5 Mpx
        if self.downsampling <= 0.0 and res > max_res:
            logging.warning(
                "Large images, downsampling to 1.5 Mpx. "
                "If this is not desired, please use --downsampling=1"
            )
            self.downsampling = (res / max_res) ** 0.5
            first_image = self._load_image(self.image_paths[0])
            self.width, self.height = first_image.shape[2], first_image.shape[1]
        self.fixed_image_size = (self.width, self.height)


        # Load COLMAP data
        self.load_colmap_data(os.path.join(args.source_path, "sparse/0"))
        self.load_anchor_data(
            args.anchors_path,
            args.num_seed_anchor_frames,
            args.anchor_submap_size,
            args.num_tail_anchor_frames,
            pose_scale=float(getattr(args, "scene_scale", 1.0)),
        )

        # Check that all images have poses
        has_all_poses = all(
            "Rt" in self.infos[image_name] for image_name in self.image_name_list
        )
        if args.use_colmap_poses:
            assert has_all_poses, (
                "COLMAP poses are required but not all images have poses."
            )
            self.align_colmap_poses()

        if args.eval_poses and not has_all_poses:
            logging.warning(
                " Not all images have COLMAP poses, pose evaluation will be skipped."
            )

        self.start_preloading()

    def __len__(self):
        return len(self.image_paths)

    @torch.no_grad()
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        info = self.infos[os.path.basename(image_path)]
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()
            image = image[:3]
        if self.mask_dir:
            mask = self._load_image(self.mask_paths[index])
            info["mask"] = mask[0][None]
        return image.cuda(), info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        if self.fixed_image_size is not None:
            if image.shape[1] != self.fixed_image_size[0] or image.shape[0] != self.fixed_image_size[1]:
                image = cv2.resize(
                    image,
                    self.fixed_image_size,
                    interpolation=cv2.INTER_AREA,
                )
        elif self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1 / self.downsampling,
                fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def _submit(self):
        if self.current_index < len(self):
            self.preload_queue.put(
                self.executor.submit(self.__getitem__, self.current_index)
            )

    def start_preloading(self):
        """Start threads to preload images."""
        for self.current_index in range(self.num_threads):
            self._submit()

    def getnext(self):
        """Get the next item from the dataset and start preloading the next one."""
        item = self.preload_queue.get().result()
        self.current_index += 1
        self._submit()
        return item

    def get_image_size(self):
        return self.height, self.width

    def load_colmap_data(self, colmap_folder_path):
        """Load COLMAP camera intrinsics and extrinsics. Stores them in self.infos."""
        has_bin = all(
            os.path.isfile(os.path.join(colmap_folder_path, name))
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        )
        has_txt = all(
            os.path.isfile(os.path.join(colmap_folder_path, name))
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        )
        if not has_bin and not has_txt:
            return
        try:
            cameras, images, _ = read_model(colmap_folder_path)
        except Exception as e:
            logging.warning(
                f" Failed to read COLMAP files in {colmap_folder_path}: {e}"
            )
            return
        if len(cameras) != 1:
            camera_list = list(cameras.values())
            ref_camera = camera_list[0]
            same_intrinsics = all(
                camera.model == ref_camera.model
                and camera.width == ref_camera.width
                and camera.height == ref_camera.height
                and np.allclose(camera.params, ref_camera.params)
                for camera in camera_list[1:]
            )
            if not same_intrinsics:
                logging.warning(
                    " Multiple camera intrinsics found; this code path assumes "
                    "a shared camera model and may be inaccurate."
                )
        model = list(cameras.values())[0].model
        if model != "PINHOLE" and model != "SIMPLE_PINHOLE":
            logging.warning(" Unexpected camera model: " + model)

        for image_id, image in images.items():
            camera = cameras[image.camera_id]

            # Intrinsics and projection matrix
            focal_x = camera.params[0]
            focal_y = camera.params[1] if camera.model == "PINHOLE" else focal_x
            focal = (focal_x + focal_y) / 2
            focal = focal_x * self.width / camera.width

            # Pose
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = qvec2rotmat(image.qvec)
            Rt[:3, 3] = image.tvec

            # Store CameraInfo for each image
            name = os.path.basename(image.name)
            if name in self.infos:
                self.infos[name]["Rt"] = torch.tensor(Rt, device="cuda")
                self.infos[name]["focal"] = torch.tensor([focal], device="cuda").float()

    def align_colmap_poses(self):
        """Scale and set first Rt as identity"""
        centres = []
        for idx in range(6):
            centres.append(self.infos[self.image_name_list[idx]]["Rt"].inverse()[:3, 3])
        centres = torch.stack(centres)
        rel_ts = centres[:-1] - centres[1:]

        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        inv_first_Rt = self.infos[self.image_name_list[0]]["Rt"].inverse()
        for info in self.infos.values():
            info["Rt"] = info["Rt"] @ inv_first_Rt
            info["Rt"][:3, 3] *= scale

    @staticmethod
    def _expected_anchor_indices_for_ranges(
        submap_ranges,
        num_seed_anchor_frames,
        num_tail_anchor_frames,
    ):
        expected_indices = set()
        for start, end in submap_ranges:
            submap_len = end - start
            front_count = min(num_seed_anchor_frames, submap_len)
            back_start = max(end - num_tail_anchor_frames, start + front_count)
            expected_indices.update(range(start, start + front_count))
            expected_indices.update(range(back_start, end))
        return expected_indices

    @staticmethod
    def _infer_anchor_submap_ranges(
        anchor_indices,
        n_images,
        num_seed_anchor_frames,
        num_tail_anchor_frames,
    ):
        if not anchor_indices or n_images == 0:
            return []
        if num_seed_anchor_frames <= 0 or num_tail_anchor_frames <= 0:
            return []

        anchor_indices = sorted(set(anchor_indices))
        anchor_index_set = set(anchor_indices)

        def has_front_anchors(start):
            end = start + num_seed_anchor_frames
            return end <= n_images and all(
                idx in anchor_index_set for idx in range(start, end)
            )

        def has_tail_anchors(end):
            start = end - num_tail_anchor_frames
            return start >= 0 and all(
                idx in anchor_index_set for idx in range(start, end)
            )

        if not has_front_anchors(0) or not has_tail_anchors(n_images):
            return []

        boundaries = [0]
        min_submap_len = num_seed_anchor_frames + num_tail_anchor_frames
        for boundary in range(1, n_images):
            if not has_tail_anchors(boundary) or not has_front_anchors(boundary):
                continue
            if boundary - boundaries[-1] < min_submap_len:
                return []
            boundaries.append(boundary)

        boundaries.append(n_images)
        submap_ranges = [
            (start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if start < end
        ]
        if not submap_ranges or any(
            end - start < min_submap_len for start, end in submap_ranges
        ):
            return []

        expected_indices = ImageDataset._expected_anchor_indices_for_ranges(
            submap_ranges,
            num_seed_anchor_frames,
            num_tail_anchor_frames,
        )
        if expected_indices != anchor_index_set:
            return []

        return submap_ranges

    @staticmethod
    def _build_fixed_anchor_submap_ranges(
        n_images,
        anchor_submap_size,
        num_seed_anchor_frames,
        num_tail_anchor_frames,
    ):
        if n_images <= 0:
            return []

        min_submap_len = num_seed_anchor_frames + num_tail_anchor_frames
        submap_ranges = []
        start = 0
        while start < n_images:
            end = min(start + anchor_submap_size, n_images)
            remaining = n_images - end
            # If the tail fragment cannot hold a valid front+back anchor layout,
            # merge it into the previous submap and use the global tail frames there.
            if remaining > 0 and remaining < min_submap_len:
                end = n_images
            submap_ranges.append((start, end))
            start = end

        return submap_ranges

    def _assign_anchor_submap_ranges(
        self,
        submap_ranges,
        num_seed_anchor_frames,
        num_tail_anchor_frames,
    ):
        self.anchor_submap_ranges = list(submap_ranges)
        for submap_id, (start, end) in enumerate(self.anchor_submap_ranges):
            submap_names = self.image_name_list[start:end]
            submap_len = len(submap_names)
            front_count = min(num_seed_anchor_frames, submap_len)
            back_start = max(submap_len - num_tail_anchor_frames, front_count)

            for image_name in submap_names:
                self.infos[image_name]["anchor_submap_id"] = submap_id

            for image_name in submap_names[:front_count]:
                self.infos[image_name]["is_seed_anchor"] = True
                self.infos[image_name]["is_submap_seed_anchor"] = True

            for image_name in submap_names[back_start:]:
                self.infos[image_name]["is_submap_tail_anchor"] = True

    def load_anchor_data(
        self,
        anchor_path,
        num_seed_anchor_frames,
        anchor_submap_size=0,
        num_tail_anchor_frames=0,
        pose_scale=1.0,
    ):
        """
        Load anchor poses from a text file.

        Expected format:
            image_name tx ty tz qw qx qy qz

        The poses are assumed to be world-to-camera.

        ``pose_scale`` multiplies tx, ty, tz on load. Rotation is untouched.
        Use this to enlarge SfM world units that are too small for the
        triangulator's absolute-depth disambiguation threshold.
        """
        pose_scale = float(pose_scale)
        if not np.isfinite(pose_scale) or pose_scale <= 0:
            raise ValueError(f"pose_scale must be a positive finite float, got {pose_scale}")
        for info in self.infos.values():
            info["is_anchor"] = False
            info["is_seed_anchor"] = False
            info["is_submap_seed_anchor"] = False
            info["is_submap_tail_anchor"] = False
            info["anchor_submap_id"] = None
        self.anchor_submap_ranges = []

        if not anchor_path:
            return

        if not os.path.exists(anchor_path):
            raise FileNotFoundError(f"Anchor pose file not found: {anchor_path}")

        if pose_scale != 1.0:
            logging.info(
                "Loading anchors from %s with scene_scale=%.6g (translations multiplied)",
                anchor_path,
                pose_scale,
            )

        ordered_anchor_names = []
        with open(anchor_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid anchor pose line with {len(parts)} fields in {anchor_path}: {line}"
                    )

                name = os.path.basename(parts[0])
                if name not in self.infos:
                    continue

                tx, ty, tz = map(float, parts[1:4])
                qw, qx, qy, qz = map(float, parts[4:8])

                Rt = np.eye(4, dtype=np.float32)
                Rt[:3, :3] = qvec2rotmat(np.array([qw, qx, qy, qz], dtype=np.float32))
                Rt[:3, 3] = (
                    np.array([tx, ty, tz], dtype=np.float32) * np.float32(pose_scale)
                )

                self.infos[name]["anchor_Rt"] = torch.tensor(Rt, device="cuda")
                self.infos[name]["is_anchor"] = True
                ordered_anchor_names.append(name)

        ordered_anchor_name_set = set(ordered_anchor_names)
        ordered_anchor_names = [
            image_name
            for image_name in self.image_name_list
            if image_name in ordered_anchor_name_set
        ]

        if anchor_submap_size <= 0:
            anchor_indices = [
                self.infos[image_name]["dataset_index"]
                for image_name in ordered_anchor_names
            ]
            submap_ranges = self._infer_anchor_submap_ranges(
                anchor_indices,
                len(self.image_name_list),
                num_seed_anchor_frames,
                num_tail_anchor_frames,
            )
            if submap_ranges:
                self._assign_anchor_submap_ranges(
                    submap_ranges,
                    num_seed_anchor_frames,
                    num_tail_anchor_frames,
                )
                logging.info(
                    "Inferred %d anchored submaps from %s: %s",
                    len(submap_ranges),
                    anchor_path,
                    ", ".join(f"{start}:{end}" for start, end in submap_ranges),
                )
                return

            if (
                num_seed_anchor_frames > 0
                and len(ordered_anchor_names) < num_seed_anchor_frames
            ):
                raise ValueError(
                    f"Requested {num_seed_anchor_frames} seed anchor frames, but "
                    f"only found {len(ordered_anchor_names)} anchors in the dataset order."
                )

            for image_name in ordered_anchor_names[:num_seed_anchor_frames]:
                self.infos[image_name]["is_seed_anchor"] = True
                self.infos[image_name]["is_submap_seed_anchor"] = True
            return

        if anchor_submap_size <= num_seed_anchor_frames:
            raise ValueError(
                "--anchor_submap_size must be larger than --num_seed_anchor_frames."
            )

        n_images = len(self.image_name_list)
        fixed_submap_ranges = self._build_fixed_anchor_submap_ranges(
            n_images,
            anchor_submap_size,
            num_seed_anchor_frames,
            num_tail_anchor_frames,
        )
        for submap_id, (start, end) in enumerate(fixed_submap_ranges):
            submap_names = self.image_name_list[start:end]
            submap_len = len(submap_names)
            front_count = min(num_seed_anchor_frames, submap_len)
            back_start = max(submap_len - num_tail_anchor_frames, front_count)
            front_names = submap_names[:front_count]
            tail_names = submap_names[back_start:]
            expected_anchor_names = list(dict.fromkeys(front_names + tail_names))
            missing_anchor_names = [
                image_name
                for image_name in expected_anchor_names
                if not self.infos[image_name]["is_anchor"]
            ]
            if missing_anchor_names:
                raise ValueError(
                    f"Missing {len(missing_anchor_names)} expected anchor frames "
                    f"for submap {submap_id} ({start}:{end}) in {anchor_path}: "
                    + ", ".join(missing_anchor_names[:8])
                )

            for image_name in front_names:
                self.infos[image_name]["is_seed_anchor"] = True
                self.infos[image_name]["is_submap_seed_anchor"] = True
            for image_name in tail_names:
                self.infos[image_name]["is_submap_tail_anchor"] = True
            for image_name in submap_names:
                self.infos[image_name]["anchor_submap_id"] = submap_id

        self.anchor_submap_ranges = fixed_submap_ranges
