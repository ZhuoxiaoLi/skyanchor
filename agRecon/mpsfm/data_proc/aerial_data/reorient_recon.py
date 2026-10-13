import numpy as np
import pycolmap

'Rotate the reconstruction to align with the plane perpendicular to the z axis.'

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return R (3x3) such that (column-vector convention): b ≈ R @ a
    """
    a = _normalize(a)
    b = _normalize(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    # a and b are (almost) parallel
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # opposite direction: rotate 180deg about any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, axis))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    # Rodrigues
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R


def reorient_reconstruction(
    recon: pycolmap.Reconstruction,
    target_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64),
) -> tuple[pycolmap.Reconstruction, np.ndarray]:
    """
    Reorient a pycolmap.Reconstruction so the estimated scene up aligns with target_up.

    Up estimation matches your torch snippet:
        up = -mean(c2w[:, :3, 1])
        up = up / ||up||

    Returns:
        (recon_reoriented, R_new_from_old)
        where x_new = R_new_from_old @ x_old (column-vector convention).
    """
    if recon is None:
        raise ValueError("recon is None")

    # Collect camera Y axes in world coordinates from c2w
    y_axes = []
    for _, image in recon.images.items():
        if not image.has_pose:
            continue

        cam_from_world = image.cam_from_world      # world -> camera (Rigid3d)
        M = np.asarray(cam_from_world.matrix(), dtype=np.float64)  # 3x4
        R_cw = M[:, :3]                              # world -> camera rotation
        R_wc = R_cw.T                                # camera -> world rotation (c2w rot)
        y_axes.append(R_wc[:, 1])                    # camera Y axis in world coords

    if len(y_axes) == 0:
        raise ValueError("No registered images with valid poses found in reconstruction.")

    up = -np.mean(np.stack(y_axes, axis=0), axis=0)
    up = _normalize(up)
    target_up = _normalize(target_up)

    # Rotation that maps up -> target_up
    R_new_from_old = _rotation_matrix_from_a_to_b(up, target_up)

    # Similarity transform (rotation only)
    sim3 = pycolmap.Sim3d(
        1.0,
        pycolmap.Rotation3d(matrix=R_new_from_old),
        np.zeros(3, dtype=np.float64),
    )

    out = pycolmap.Reconstruction(recon)
    out.transform(sim3)
    return out, R_new_from_old

if __name__ == "__main__":
    recon = pycolmap.Reconstruction("/home/dsxy/Project/agMatching/data/horizongs/road/a2g-test/sparse-aerial")
    recon_, _ = reorient_reconstruction(recon)
    recon_.write("/home/dsxy/Project/agMatching/data/horizongs/road/a2g-test/temp")
