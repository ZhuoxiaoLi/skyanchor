"""Dataset and parser for reconstructing user provided images."""

from pathlib import Path

import pycolmap
import torch
import yaml
from PIL import Image
from pycolmap import Reconstruction
import numpy as np

from .basedataset import BaseDataset, BaseDatasetParser


class AGDataset(BaseDataset, torch.utils.data.Dataset):
    """Dataset class for user provided images following standard pipeline structure."""

    def _init(self, image_list=None, scene_parser=None, images_dir=None, **kwargs):
        assert self.conf.batch_size == 1, "Batch size must be 1 for this dataset"
        if scene_parser is None:
            assert images_dir is not None, "images_dir must be provided if scene_parser is None"
            self.scene_parser = AGParser(rgb_dir=images_dir.parent, imnames=image_list)
        else:
            self.scene_parser = scene_parser
        if image_list is None:
            image_ids = list(self.scene_parser.rec.images.keys())
        else:
            image_ids = [k for k, v in self.scene_parser.rec.images.items() if v.name in image_list]

        image_ids.sort()
        self.imids = image_ids

class AGParser(BaseDatasetParser):
    """Parser for user provided images following standard pipeline structure."""

    scene = "<custom>"
    dataset = AGDataset

    def __init__(self, *args, intrinsics_pth=None, recon_exist_dir=None, rgb_register_dir=None, rgb_unregister_dir=None, submap_im_num=50, anchor_num_pair=(6,6), **kwargs):
        self.rec = Reconstruction(recon_exist_dir)
        self.reconstruction_dir = None

        self.rec_origin = Reconstruction(self.rec)
        self.rec_out = Reconstruction(self.rec)

        max_exist_image_id = max(self.rec.images.keys())
        max_exist_camera_id = max(self.rec.cameras.keys())

        self.anchor_num_pair = anchor_num_pair

        self.imnames2path = {}
        self.imnames_register = []
        self.imnames_unregister = []

        self.imnames2dir = {}

        rgb_register_dir = Path(rgb_register_dir)
        rgb_unregister_dir = Path(rgb_unregister_dir)

        registered_names = {image.name for image in self.rec.images.values() if image.has_pose}
        for p in sorted(rgb_register_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}:
                name = str(p.relative_to(rgb_register_dir))
                if name not in registered_names:
                    continue
                self.imnames_register.append(name)
                self.imnames2path[name] = p
                self.imnames2dir[name] = rgb_register_dir

        missing_references = registered_names - set(self.imnames_register)
        if missing_references:
            raise FileNotFoundError(
                f"Missing {len(missing_references)} aerial reference images: "
                f"{sorted(missing_references)[:10]}"
            )

        # for p in sorted(rgb_unregister_dir.rglob('*')):
        #     if p.is_file():
        #         self.imnames_unregister.append(name)
        #         self.imnames2path[name] = p
        #         self.imnames2dir[name] = rgb_unregister_dir

        self.collect_unregister_images(
            rgb_unregister_dir=rgb_unregister_dir,
            subnum=submap_im_num,
            anchor_pair=anchor_num_pair
        )

        self.imnames = self.imnames_register + self.imnames_unregister


        intrinsics_pth = Path(intrinsics_pth)
        with open(intrinsics_pth, encoding="utf-8") as f:
            intrinsics = yaml.safe_load(f)

        for camid, camdict in intrinsics.items():
            params = camdict["params"]
            images = camdict["images"]
            if images == "all":
                images = self.imnames_unregister
            else:
                images = [imname for imname in images if imname in self.imnames2path]

            shapes = []
            for imname in images:
                im = Image.open(self.imnames2path[imname])
                shapes.append(im.size)
            if shapes:
                assert len(set(shapes)) == 1, "All images must have the same shape"
            else:
                continue
            # assert len(set(shapes)) == 1, "All images must have the same shape"
            width, height = shapes[0]

            max_exist_camera_id += 1
            camid = max_exist_camera_id
            camera = pycolmap.Camera(
                model="PINHOLE",
                width=width,
                height=height,
                params=params,
                has_prior_focal_length=True,
                camera_id=camid,
            )
            self.rec.add_camera(camera)
            self.rec_out.add_camera(camera)

            for imname in images:
                max_exist_image_id += 1
                image_id = max_exist_image_id
                self.rec.add_image(pycolmap.Image(name=imname, camera_id=camid, image_id=image_id))

        name2imid = {img.name: imid for imid, img in self.rec.images.items()}
        self.name2imid = name2imid

        self.imids_register = [name2imid[name] for name in self.imnames_register]
        self.imids_unregister = [name2imid[name] for name in self.imnames_unregister]

    def rgb(self, imid):
        """Get RGB image for the image id."""
        imname = self.image_name(imid)
        im = Image.open(self.imnames2path[imname])
        im = np.array(im)
        return im

    def update_register(self):
        if not self.imids_unregister:
            return False  # nothing to move

        first_imid = self.imids_unregister.pop(0)
        self.imids_register.append(first_imid)
        return True

    def update_recon_out(self, src_recon: Reconstruction, ref_image_id: int):
        src_image = src_recon.image(ref_image_id)

        # Output image will contain only 2D points that originally had a 3D association
        new_points2D = pycolmap.Point2DList()
        src_point3D_ids_for_new_points2D = []  # maps new point2D idx -> source point3D_id

        for src_point2D in src_image.points2D:
            if src_point2D.has_point3D():
                new_points2D.append(
                    pycolmap.Point2D(xy=src_point2D.xy, point3D_id=pycolmap.INVALID_POINT3D_ID)
                )
                src_point3D_ids_for_new_points2D.append(int(src_point2D.point3D_id))

        new_image = pycolmap.Image(
            name=src_image.name,
            camera_id=src_image.camera_id,
            image_id=src_image.image_id,  # if conflict risk, omit and let it auto-assign
            points2D=new_points2D,
            cam_from_world=src_image.cam_from_world,
        )

        self.rec_out.add_image(new_image)
        out_ref_image = self.rec_out.image(new_image.image_id)

        # For each kept 2D point in the reference image, recreate its corresponding 3D point and track
        for new_point2D_idx, src_point3D_id in enumerate(src_point3D_ids_for_new_points2D):
            src_point3D = src_recon.point3D(src_point3D_id)
            src_track_elements = src_point3D.track.elements

            out_track = pycolmap.Track()
            pending_point2D_assignments = []  # list[(out_image_ref, out_point2D_idx)]

            for track_el in src_track_elements:
                track_image_id = int(track_el.image_id)
                track_point2D_idx_in_src = int(track_el.point2D_idx)

                if track_image_id == ref_image_id:
                    # point2D already exists in out_ref_image at new_point2D_idx
                    out_track.add_element(track_image_id, new_point2D_idx)
                    pending_point2D_assignments.append((out_ref_image, new_point2D_idx))
                else:
                    src_other_image = src_recon.image(track_image_id)
                    src_other_point2D = src_other_image.point2D(track_point2D_idx_in_src)

                    out_other_image = self.rec_out.image(track_image_id)
                    out_other_image.points2D.append(
                        pycolmap.Point2D(xy=src_other_point2D.xy, point3D_id=pycolmap.INVALID_POINT3D_ID)
                    )
                    out_point2D_idx = len(out_other_image.points2D) - 1

                    out_track.add_element(track_image_id, out_point2D_idx)
                    pending_point2D_assignments.append((out_other_image, out_point2D_idx))

            out_point3D_id = self.rec_out.add_point3D(
                xyz=src_point3D.xyz,
                track=out_track,
                color=src_point3D.color,
            )

            for out_image_ref, out_point2D_idx in pending_point2D_assignments:
                out_image_ref.set_point3D_for_point2D(int(out_point2D_idx), int(out_point3D_id))

    def update_recon_out_list(self, src_recon: Reconstruction, ref_image_ids: list):
        add_point3d_ids = set()
        src_point3D_ids_for_new_points2D_all = {}

        for ref_image_id in ref_image_ids:
            src_image = src_recon.image(ref_image_id)

            # Output image will contain only 2D points that originally had a 3D association
            new_points2D = pycolmap.Point2DList()
            src_point3D_ids_for_new_points2D = []  # maps new point2D idx -> source point3D_id
            src_point3D_ids_for_new_points2D_ = {}

            iterr = 0
            for src_point2D in src_image.points2D:
                if src_point2D.has_point3D():
                    new_points2D.append(
                        pycolmap.Point2D(xy=src_point2D.xy, point3D_id=pycolmap.INVALID_POINT3D_ID)
                    )
                    src_point3D_ids_for_new_points2D.append(int(src_point2D.point3D_id))
                    src_point3D_ids_for_new_points2D_[int(src_point2D.point3D_id)] = iterr
                    add_point3d_ids.add(int(src_point2D.point3D_id))
                    iterr += 1

            src_point3D_ids_for_new_points2D_all[ref_image_id] = src_point3D_ids_for_new_points2D_

            new_image = pycolmap.Image(
                name=src_image.name,
                camera_id=src_image.camera_id,
                image_id=src_image.image_id,  # if conflict risk, omit and let it auto-assign
                points2D=new_points2D,
                cam_from_world=src_image.cam_from_world,
            )

            self.rec_out.add_image(new_image)

        aaa = 0
        for point3d_ids in add_point3d_ids:

            src_point3D = src_recon.point3D(point3d_ids)
            src_track_elements = src_point3D.track.elements

            out_track = pycolmap.Track()
            pending_point2D_assignments = []  # list[(out_image_ref, out_point2D_idx)]

            for track_el in src_track_elements:
                track_image_id = int(track_el.image_id)

                if track_image_id in ref_image_ids:
                    # point2D already exists in out_ref_image at new_point2D_idx
                    src_point3D_ids_for_new_points2D = src_point3D_ids_for_new_points2D_all[track_image_id]
                    new_point2D_idx = src_point3D_ids_for_new_points2D[point3d_ids]

                    out_track.add_element(track_image_id, new_point2D_idx)
                    pending_point2D_assignments.append((self.rec_out.image(track_image_id), new_point2D_idx))
                else:
                    track_point2D_idx_in_src = int(track_el.point2D_idx)
                    src_other_image = src_recon.image(track_image_id)
                    src_other_point2D = src_other_image.point2D(track_point2D_idx_in_src)

                    out_other_image = self.rec_out.image(track_image_id)
                    out_other_image.points2D.append(
                        pycolmap.Point2D(xy=src_other_point2D.xy, point3D_id=pycolmap.INVALID_POINT3D_ID)
                    )
                    out_point2D_idx = len(out_other_image.points2D) - 1

                    out_track.add_element(track_image_id, out_point2D_idx)
                    pending_point2D_assignments.append((out_other_image, out_point2D_idx))

            for el in out_track.elements:
                img = self.rec_out.image(int(el.image_id))
                p2d = img.point2D(int(el.point2D_idx))
                if p2d.has_point3D():
                    print("[HasPoint3D conflict]",
                          "image_id=", int(el.image_id),
                          "point2D_idx=", int(el.point2D_idx),
                          "existing_point3D_id=", int(p2d.point3D_id),
                          "img_points2D_len=", len(img.points2D))
                    break

            seen = {}  # key -> first_index
            dups = []  # (key, first_i, curr_i)

            for i, el in enumerate(out_track.elements):
                key = (int(el.image_id), int(el.point2D_idx))
                if key in seen:
                    dups.append((key, seen[key], i))
                else:
                    seen[key] = i
            if dups:
                continue

            try:
                out_point3D_id = self.rec_out.add_point3D(
                    xyz=src_point3D.xyz,
                    track=out_track,
                    color=src_point3D.color,
                )
            except Exception as e:
                print("[add_point3D ERROR]", repr(e))
                print("[out_track dump] elements_len =", len(out_track.elements))

                for i, el in enumerate(out_track.elements):
                    img_id = int(el.image_id)
                    p2d_idx = int(el.point2D_idx)

                    print(f"  - el[{i}] image_id={img_id} point2D_idx={p2d_idx}")

                    try:
                        img = self.rec_out.image(img_id)
                        print(f"    img_points2D_len={len(img.points2D)} name={getattr(img, 'name', None)}")
                    except Exception as e_img:
                        print("    [image fetch failed]", repr(e_img))
                        continue

                    try:
                        p2d = img.point2D(p2d_idx)
                        has3d = p2d.has_point3D()
                        print(
                            f"    p2d_has_point3D={has3d} existing_point3D_id={int(p2d.point3D_id) if has3d else None} xy={p2d.xy}")
                    except Exception as e_p2d:
                        print("    [point2D fetch failed]", repr(e_p2d))

                raise

            for out_image_ref, out_point2D_idx in pending_point2D_assignments:
                out_image_ref.set_point3D_for_point2D(int(out_point2D_idx), int(out_point3D_id))

            aaa += 1

    def collect_unregister_images(self, rgb_unregister_dir, subnum, anchor_pair=(3, 4)):
        all_files = sorted([p for p in rgb_unregister_dir.rglob('*')
                            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        all_names = [str(p.relative_to(rgb_unregister_dir)) for p in all_files]

        front_n, back_n = anchor_pair
        anchor_total = front_n + back_n
        total_num = len(all_names)

        self.imnames_unregister = []

        if total_num < anchor_total:
            raise ValueError(
                f"Insufficient images in rgb_unregister_dir: found {total_num} images; "
                f"anchor_pair={anchor_pair} requires at least {anchor_total} images."
            )

        def add_selected(names):
            for name in names:
                self.imnames_unregister.append(name)
                self.imnames2path[name] = rgb_unregister_dir / name
                self.imnames2dir[name] = rgb_unregister_dir

        if total_num <= subnum:
            selected = all_names[:front_n] + all_names[-back_n:]
            add_selected(selected)
            return

        remainder = total_num % subnum
        full_group_count = total_num // subnum
        groups = []

        if remainder == 0:
            for i in range(full_group_count):
                groups.append(all_names[i * subnum:(i + 1) * subnum])

        elif remainder < subnum * 0.5:
            for i in range(full_group_count - 1):
                groups.append(all_names[i * subnum:(i + 1) * subnum])
            groups.append(all_names[(full_group_count - 1) * subnum:])

        else:
            for i in range(full_group_count):
                groups.append(all_names[i * subnum:(i + 1) * subnum])
            groups.append(all_names[full_group_count * subnum:])

        for idx, group in enumerate(groups):
            if len(group) < anchor_total:
                raise ValueError(
                    f"Group {idx} has insufficient images: found {len(group)} images; "
                    f"anchor_pair={anchor_pair} requires at least {anchor_total} images."
                )

            selected = group[:front_n] + group[-back_n:]
            add_selected(selected)
