"""I/O functions for reading and writing data. Many copied from and inspired by hloc."""

from pathlib import Path

import cv2
import h5py
import numpy as np

from mpsfm.utils.parsers import names_to_pair, names_to_pair_old, read_unique_pairs


def get_mono_map(path, name):
    with h5py.File(str(path), "r") as f:
        return {k: v[:] for k, v in f[str(name)].items()}


def get_mono_map_from_pairs(path, name, pairs_path):
    pairs = read_unique_pairs(pairs_path)
    cname = str(name)
    with h5py.File(str(path), "r") as f:
        depths = []
        valids = []
        variances = []
        scores = []
        for pair in pairs:
            if cname not in pair:
                continue
            key = f"{names_to_pair(*pair)}/{cname}"
            depths.append(f[key]["depth"][:])
            valids.append(f[key]["valid"][:])
            variances.append(f[key]["variance"][:])
            scores.append(((1 / variances[-1])[valids[-1]]).mean())

        if len(scores) == 0:
            return None
        amax = np.argmax(scores)

        return {
            "depth": depths[amax],
            "valid": valids[amax],
            "depth_variance": variances[amax],
        }


def get_mask(path, name):
    with h5py.File(path, "r") as file:
        return file[name]["mask"][:]


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(path: Path, name: str, return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(f"Could not find pair {(name0, name1)}... " "Maybe you matched with a different list of pairs? ")


def get_dense_2view_keypoints(path: Path, name0: str, name1: str):
    with h5py.File(path, "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        kps0 = hfile[pair][name0]["keypoints"].__array__()
        kps1 = hfile[pair][name1]["keypoints"].__array__()
    return kps0, kps1


def get_matches(path: Path, name0: str, name1: str) -> tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores
