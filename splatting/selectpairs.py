"""Select covisible views for RoMa, preserving real COLMAP image IDs."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import numpy as np
from colmap_io import read_model


def select_pairs(model, output, max_views=5, workers=8):
    if max_views < 1 or workers < 1:
        raise ValueError('max_views and workers must be positive')
    result = read_model(str(model))
    if result is None:
        raise ValueError(f'No COLMAP model in {model}')
    _, images, points = result
    ids = sorted(images)
    centers = {i: -images[i].qvec2rotmat().T @ images[i].tvec for i in ids}
    tracks = {i: np.unique(images[i].point3D_ids[images[i].point3D_ids >= 0]) for i in ids}
    neighbors = {i: [] for i in ids}

    def score(pair):
        i, j = pair
        common = np.intersect1d(tracks[i], tracks[j], assume_unique=True)
        if not len(common):
            return i, j, 0
        xyz = np.array([points[int(p)].xyz for p in common])
        a, b = centers[i] - xyz, centers[j] - xyz
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        valid = den > 1e-12
        if not valid.any():
            return i, j, 0
        angles = np.degrees(np.arccos(np.clip(np.sum(a[valid]*b[valid], axis=1)/den[valid], -1, 1)))
        angle = np.sort(angles)[int(len(angles)*0.75)]
        return i, j, len(angles) if angle >= 1 else 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, j, value in pool.map(score, combinations(ids, 2)):
            if value > 0:
                neighbors[i].append((j, value))
                neighbors[j].append((i, value))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        f.write(f'{len(ids)}\n')
        for i in ids:
            top = sorted(neighbors[i], key=lambda item: (-item[1], item[0]))[:max_views]
            f.write(f'{i}\n{len(top)} ' + ' '.join(f'{j} {v}' for j,v in top) + '\n')
    print(f'[Pairs] {len(ids)} cameras, {sum(bool(v) for v in neighbors.values())} with covisible neighbors -> {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dense_folder', '-i', required=True, help='COLMAP model directory')
    parser.add_argument('--save_folder', '-o', required=True)
    parser.add_argument('--max_d', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    select_pairs(args.dense_folder, Path(args.save_folder)/'pair.txt', args.max_d, args.workers)
