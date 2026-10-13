from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
import pycolmap


@dataclass(frozen=True)
class ImageKDIndex:
    """KDTree index for nearest-neighbor queries by image name."""
    names: List[str]                 # registered image names, length N
    centers: np.ndarray              # shape (N, 3)
    name_to_idx: Dict[str, int]      # name -> index in names/centers
    kdtree: cKDTree                  # KDTree built on centers


@dataclass(frozen=True)
class FolderKDIndex:
    """Group KDTree indices by folder. by_folder maps folders to ImageKDIndex; name_to_folder maps full image names to folders."""
    by_folder: Dict[str, ImageKDIndex]
    name_to_folder: Dict[str, str]


def _folder_key(image_name: str) -> str:
    """Return the parent folder of a reconstruction image name, or "." for a basename."""
    p = Path(image_name)
    parent = p.parent.as_posix()
    return parent if parent not in ("", ".") else "."


def build_image_kdtree(recon: pycolmap.Reconstruction) -> FolderKDIndex:
    """Build one camera-center KDTree per folder from registered reconstruction images."""
    folder_to_names: Dict[str, List[str]] = {}
    folder_to_centers: Dict[str, List[np.ndarray]] = {}
    name_to_folder: Dict[str, str] = {}

    for _, im in recon.images.items():
        if not im.has_pose:
            continue

        C = im.cam_from_world.inverse().translation  # camera center in world coords
        name = im.name                               # e.g. 'z/1.jpg'
        folder = _folder_key(name)

        folder_to_names.setdefault(folder, []).append(name)
        folder_to_centers.setdefault(folder, []).append(np.asarray(C, dtype=np.float64).reshape(3,))
        name_to_folder[name] = folder

    if not name_to_folder:
        raise ValueError("No registered images with valid poses found in reconstruction.")

    by_folder: Dict[str, ImageKDIndex] = {}
    for folder, names in folder_to_names.items():
        centers_arr = np.stack(folder_to_centers[folder], axis=0)  # (N, 3)
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        by_folder[folder] = ImageKDIndex(
            names=names,
            centers=centers_arr,
            name_to_idx=name_to_idx,
            kdtree=cKDTree(centers_arr),
        )

    return FolderKDIndex(by_folder=by_folder, name_to_folder=name_to_folder)


def query_nearest_images_by_names_from_kdtree(
    sfm_pairs_path: Union[str, Path],
    index: FolderKDIndex,
    init_pair: Tuple[str, str],
    n: int,
):
    """Use the registered member of init_pair as the query. Retrieve n nearest images from its folder, excluding itself and optionally including the other same-folder pair member. Return neighbor names and the output pair-file path."""
    name1, name2 = init_pair

    if name1 in index.name_to_folder:
        query_name, other_name = name1, name2
    elif name2 in index.name_to_folder:
        query_name, other_name = name2, name1
    else:
        raise KeyError(f"Unknown image pair (neither registered in index): {init_pair}")

    folder = index.name_to_folder[query_name]
    folder_index = index.by_folder[folder]

    Nf = len(folder_index.names)
    q_idx = folder_index.name_to_idx[query_name]
    q = folder_index.centers[q_idx]

    k = min(Nf, n)
    dists, inds = folder_index.kdtree.query(q, k=k)

    inds = np.atleast_1d(inds)

    inds = inds[:n]

    inds_names = [folder_index.names[i] for i in inds]

    inds_names.append(other_name)

    group_pairs = list(combinations(inds_names, 2))

    sfm_pairs_path = Path(sfm_pairs_path)
    new_path = sfm_pairs_path.with_name(f"{sfm_pairs_path.stem}-loca{sfm_pairs_path.suffix}")

    # with open(new_path, "r", encoding="utf-8") as f:
    #     inds_names = {name for line in f for name in line.split()}
    #
    # inds_names = list(inds_names)
    #
    with open(new_path, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in group_pairs) + "\n")

    return inds_names, new_path, group_pairs
