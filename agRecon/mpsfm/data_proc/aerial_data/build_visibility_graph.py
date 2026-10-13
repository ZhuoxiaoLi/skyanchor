from .reorient_recon import reorient_reconstruction

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import pycolmap
import numpy as np

@dataclass(frozen=True)
class Box2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def is_valid(self) -> bool:
        return self.xmax > self.xmin and self.ymax > self.ymin

    def area(self) -> float:
        if not self.is_valid():
            return 0.0
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def intersect(self, other: "Box2D") -> "Box2D":
        return Box2D(
            xmin=max(self.xmin, other.xmin),
            ymin=max(self.ymin, other.ymin),
            xmax=min(self.xmax, other.xmax),
            ymax=min(self.ymax, other.ymax),
        )


def compute_image_box_from_points3d_xy(
    recon: pycolmap.Reconstruction,
    image_id: int,
) -> Optional[Box2D]:
    """Return the 2D bounding box of observations with valid point3D IDs for image_id, or None if no valid observations exist."""
    img = recon.images[image_id]
    xs, ys = [], []

    for p2d in img.points2D:
        if p2d.point3D_id == -1:
            continue
        x, y = float(p2d.xy[0]), float(p2d.xy[1])
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    if not xs:
        return None

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    box = Box2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return box if box.is_valid() else None


def compute_image_box_from_points3d_xy_robust(
    recon: pycolmap.Reconstruction,
    image_id: int,
    q_low: float = 0.02,
    q_high: float = 0.98,
    min_points: int = 30,
) -> Optional[Box2D]:
    """Compute an observation bounding box after quantile trimming between q_low and q_high. Use points with valid point3D IDs; skip trimming below min_points to avoid unstable quantiles."""
    img = recon.images[image_id]
    xs, ys = [], []

    for p2d in img.points2D:
        pid = int(p2d.point3D_id)
        if pid == -1:
            continue
        if pid not in recon.points3D:
            continue

        p3d = recon.points3D[pid]
        x, y = float(p3d.xyz[0]), float(p3d.xyz[1])
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    if len(xs) == 0:
        return None

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if len(xs) >= min_points:
        xmin = float(np.quantile(xs, q_low))
        xmax = float(np.quantile(xs, q_high))
        ymin = float(np.quantile(ys, q_low))
        ymax = float(np.quantile(ys, q_high))
    else:
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())

    box = Box2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return box if box.is_valid() else None

def box_iou(a: Optional[Box2D], b: Optional[Box2D]) -> float:
    """Return intersection-over-union area, or zero for missing or invalid boxes."""
    if a is None or b is None or (not a.is_valid()) or (not b.is_valid()):
        return 0.0

    inter = a.intersect(b)
    inter_area = inter.area()
    if inter_area <= 0.0:
        return 0.0

    union = a.area() + b.area() - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union

def build_overlap_graph_topn(
    recon: pycolmap.Reconstruction,
    top_n: int = 6,
) -> Tuple[Dict[int, Optional[Box2D]], Dict[int, List[Tuple[int, float]]]]:
    """Return per-image boxes and a graph mapping each image ID to at most top_n (neighbor, overlap) entries sorted by decreasing overlap. Pairwise construction is O(M^2) for M images."""
    recon_reori, _ = reorient_reconstruction(recon)

    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    image_ids = list(recon_reori.images.keys())

    boxes: Dict[int, Optional[Box2D]] = {}
    for i in image_ids:
        boxes[i] = compute_image_box_from_points3d_xy_robust(recon_reori, i)

    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in image_ids}

    for idx_i, i in enumerate(image_ids):
        bi = boxes[i]
        if bi is None:
            continue

        overlaps: List[Tuple[int, float]] = []
        for idx_j, j in enumerate(image_ids):
            if j == i:
                continue
            bj = boxes[j]
            s = box_iou(bi, bj)
            if s > 0.0:
                overlaps.append((j, s))

        overlaps.sort(key=lambda t: t[1], reverse=True)
        graph[i] = overlaps[:top_n]

    return boxes, graph

def build_overlap_graph_topn_name(
    recon: pycolmap.Reconstruction,
    top_n: int = 6,
) -> Tuple[Dict[str, Optional[Box2D]], Dict[str, List[Tuple[str, float]]]]:
    """Return boxes and an overlap graph keyed by image names. Each graph entry holds up to top_n (neighbor_name, overlap) pairs sorted by decreasing overlap."""
    recon_reori, _ = reorient_reconstruction(recon)

    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    image_ids = list(recon_reori.images.keys())

    id2name: Dict[int, str] = {i: recon_reori.images[i].name for i in image_ids}

    boxes: Dict[str, Optional[Box2D]] = {}
    for i in image_ids:
        name = id2name[i]
        boxes[name] = compute_image_box_from_points3d_xy_robust(recon_reori, i)

    graph: Dict[str, List[Tuple[str, float]]] = {id2name[i]: [] for i in image_ids}

    for i in image_ids:
        name_i = id2name[i]
        bi = boxes[name_i]
        if bi is None:
            continue

        overlaps: List[Tuple[str, float]] = []
        for j in image_ids:
            if j == i:
                continue
            name_j = id2name[j]
            bj = boxes[name_j]
            s = box_iou(bi, bj)
            if s > 0.0:
                overlaps.append((name_j, s))

        overlaps.sort(key=lambda t: t[1], reverse=True)
        graph[name_i] = overlaps[:top_n]

    return boxes, graph


if __name__ == "__main__":
    recon = pycolmap.Reconstruction("/home/dsxy/Project/agMatching/data/horizongs/road/a2g-test/sparse-aerial")
    boxes, graph = build_overlap_graph_topn_name(recon, 10)


