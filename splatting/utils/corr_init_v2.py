"""Initialize dense geometry with RoMaV2 correspondences. to_pixel_coordinates takes (H_A, W_A, H_B, W_B) and returns (x, y) coordinates. match returns a dictionary containing warp_AB and overlap_AB."""
import sys
import os

from matplotlib import pyplot as plt
from PIL import Image
import torch
torch.set_float32_matmul_precision('highest')
import numpy as np
import cv2

from tqdm import tqdm
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist

import torch.nn.functional as F
from plyfile import PlyData, PlyElement



def get_camera_alpha_mask_np(camera):
    alpha_mask = getattr(camera, "alpha_mask", None)
    if alpha_mask is None:
        alpha_mask = getattr(camera, "gt_alpha_mask", None)
    if alpha_mask is None:
        return None

    alpha_np = alpha_mask.detach().cpu().float().numpy()
    if alpha_np.ndim == 3:
        alpha_np = alpha_np[0]
    return (alpha_np >= 0.5).astype(np.float32)


def read_pair_file(pair_path, num_src_views=10):
    """Read matching neighbors from pair.txt using COLMAP image IDs. The first line stores the image count; each image then has an ID line and a num_pairs/src_id/score line. Keep at most num_src_views neighbors and return (pair_dict, num_images)."""
    pair_dict = {}

    with open(pair_path, 'r') as f:
        num_images = int(f.readline().strip())

        while True:
            line1 = f.readline()
            if not line1:
                break

            ref_colmap_id = int(line1.strip())

            line2 = f.readline().strip().split()
            num_pairs = int(line2[0])

            src_colmap_ids = []
            for i in range(min(num_pairs, num_src_views)):
                src_colmap_id = int(line2[1 + i * 2])
                src_colmap_ids.append(src_colmap_id)

            pair_dict[ref_colmap_id] = src_colmap_ids

    print(f"[Pair File] Loaded {len(pair_dict)} camera pairs from {pair_path}")
    return pair_dict, num_images

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'submodules', 'RoMaV2', 'src'))
from romav2 import RoMaV2


def pairwise_distances(matrix):
    """
    Computes the pairwise Euclidean distances between all vectors in the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N, N].
    """
    # Compute squared pairwise distances
    squared_diff = torch.cdist(matrix, matrix, p=2)
    return squared_diff


def k_closest_vectors(matrix, k):
    """
    Finds the k-closest vectors for each vector in the input matrix based on Euclidean distance.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.
        k (int): Number of closest vectors to return for each vector.

    Returns:
        torch.Tensor: Indices of the k-closest vectors for each vector, excluding the vector itself.
    """
    # Compute pairwise distances
    distances = pairwise_distances(matrix)

    # For each vector, sort distances and get the indices of the k-closest vectors (excluding itself)
    # Set diagonal distances to infinity to exclude the vector itself from the nearest neighbors
    distances.fill_diagonal_(float('inf'))

    # Get the indices of the k smallest distances (k-closest vectors)
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    return indices


def select_cameras_kmeans(cameras, K):
    """
    Selects up to K cameras from a set using K-means clustering.

    Args:
        cameras: NumPy array of shape (N, 16), representing N cameras with their 4x4 homogeneous matrices flattened.
        K: Number of clusters (cameras to select).

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
            May contain fewer than K entries when scipy drops empty clusters
            (common when many input cameras have near-duplicate poses).
    """
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)

    if cameras.shape[1] != 16:
        raise ValueError("Each camera must have 16 values corresponding to a flattened 4x4 matrix.")

    N = cameras.shape[0]
    if N == 0:
        return []
    K_eff = int(min(max(K, 1), N))

    # scipy.cluster.vq.kmeans silently drops empty clusters, so it can return
    # fewer than K_eff centroids when many cameras share near-duplicate poses
    # (aerial captures, video frames, etc.), causing `for k in range(K)` to
    # index out of bounds on cluster_centers. Prefer kmeans2 with k-means++
    # init which is more robust; fall back to the legacy kmeans if unavailable.
    cluster_centers = None
    cluster_assignments = None
    try:
        from scipy.cluster.vq import kmeans2
        cluster_centers, cluster_assignments = kmeans2(
            cameras, K_eff, minit='++', missing='warn', seed=0
        )
    except Exception as exc:
        print(f"[K-means] kmeans2 failed ({exc!r}); falling back to legacy kmeans.")
        cluster_centers, _ = kmeans(cameras, K_eff)
        cluster_assignments, _ = vq(cameras, cluster_centers)

    actual_K = int(cluster_centers.shape[0])
    if actual_K < K_eff:
        print(f"[K-means] Warning: requested {K_eff} clusters but only {actual_K} non-empty "
              f"clusters were produced (likely duplicate/near-duplicate camera poses).")

    selected_indices = []
    used = set()
    for k in range(actual_K):
        members_mask = (cluster_assignments == k)
        member_idx = np.where(members_mask)[0]
        if member_idx.size == 0:
            continue
        cluster_members = cameras[member_idx]
        distances = cdist(cluster_centers[k:k + 1], cluster_members)[0]
        nearest_camera_idx = int(member_idx[np.argmin(distances)])
        if nearest_camera_idx in used:
            continue
        used.add(nearest_camera_idx)
        selected_indices.append(nearest_camera_idx)

    # If duplicate or empty clusters left us short, top up with a greedy
    # farthest-point pass over the remaining cameras so we still return up to
    # K_eff distinct references whenever possible.
    if len(selected_indices) < K_eff and len(used) < N:
        remaining = [i for i in range(N) if i not in used]
        sel_arr = cameras[selected_indices] if selected_indices else None
        while len(selected_indices) < K_eff and remaining:
            rem_arr = cameras[remaining]
            if sel_arr is None or len(sel_arr) == 0:
                pick = 0
            else:
                d = cdist(rem_arr, sel_arr).min(axis=1)
                pick = int(np.argmax(d))
            chosen = remaining.pop(pick)
            selected_indices.append(chosen)
            used.add(chosen)
            sel_arr = cameras[selected_indices]

    return selected_indices


def compute_warp_and_confidence_v2(viewpoint_cam1, viewpoint_cam2, roma_model, device="cuda", verbose=False, output_dict={}):
    """Compute RoMaV2 correspondences between two cameras. Return certainty (H, W), normalized warp (H, W, 4) in [x_A, y_A, x_B, y_B] order, and the processed NumPy target image imB. verbose enables visualization."""
    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
    imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        cax1 = ax[0].imshow(imA)
        ax[0].set_title("Image 1")
        cax2 = ax[1].imshow(imB)
        ax[1].set_title("Image 2")
        fig.colorbar(cax1, ax=ax[0])
        fig.colorbar(cax2, ax=ax[1])

        for axis in ax:
            axis.axis('off')
        # Save the figure into the dictionary
        output_dict[f'image_pair'] = fig

    preds = roma_model.match(imA, imB)

    warp_AB = preds["warp_AB"]
    overlap_AB = preds["overlap_AB"]  # shape: (1, H, W, 1)

    B, H, W, _ = warp_AB.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1 + 1/H, 1 - 1/H, H, device=device),
        torch.linspace(-1 + 1/W, 1 - 1/W, W, device=device),
        indexing='ij'
    )
    im_A_coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    warp = torch.cat([im_A_coords, warp_AB], dim=-1)  # (1, H, W, 4)

    certainty = overlap_AB[0, :, :, 0]  # (H, W)

    return certainty, warp[0], np.array(imB)


def resize_batch(tensors_3d, tensors_4d, target_shape):
    """
    Resizes a batch of tensors with shapes [B, H, W] and [B, H, W, 4] to the target spatial dimensions.

    Args:
        tensors_3d: Tensor of shape [B, H, W].
        tensors_4d: Tensor of shape [B, H, W, 4].
        target_shape: Tuple (target_H, target_W) specifying the target spatial dimensions.

    Returns:
        resized_tensors_3d: Tensor of shape [B, target_H, target_W].
        resized_tensors_4d: Tensor of shape [B, target_H, target_W, 4].
    """
    target_H, target_W = target_shape

    # Resize [B, H, W] tensor
    resized_tensors_3d = F.interpolate(
        tensors_3d.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).squeeze(1)

    # Resize [B, H, W, 4] tensor
    B, _, _, C = tensors_4d.shape
    resized_tensors_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    return resized_tensors_3d, resized_tensors_4d


def aggregate_confidences_and_warps_v2(viewpoint_stack, closest_indices, roma_model, source_idx, verbose=False, output_dict={}):
    """Aggregate confidence and warps over source_idx neighbors. Return maximum confidences, associated warps, per-pixel neighbor indices, source image imA, and target images imB_compound."""
    certainties_all, warps_all, imB_compound = [], [], []

    for nn in (closest_indices[source_idx]):

        viewpoint_cam1 = viewpoint_stack[source_idx]
        viewpoint_cam2 = viewpoint_stack[nn]

        certainty, warp, imB = compute_warp_and_confidence_v2(viewpoint_cam1, viewpoint_cam2, roma_model, verbose=verbose, output_dict=output_dict)
        certainties_all.append(certainty)
        warps_all.append(warp)
        imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)
    target_shape = imB_compound[0].shape[:2]
    if verbose:
        print("certainties_all.shape:", certainties_all.shape)
        print("torch.stack(warps_all, dim=0).shape:", torch.stack(warps_all, dim=0).shape)
        print("target_shape:", target_shape)

    certainties_all_resized, warps_all_resized = resize_batch(certainties_all,
                                                              torch.stack(warps_all, dim=0),
                                                              target_shape
                                                              )

    if verbose:
        print("warps_all_resized.shape:", warps_all_resized.shape)
        for n, cert in enumerate(certainties_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence")
            output_dict[f'certainty_{n}'] = fig

        for n, warp in enumerate(warps_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp")
            output_dict[f'warp_resized_{n}'] = fig

        for n, cert in enumerate(certainties_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence resized")
            output_dict[f'certainty_resized_{n}'] = fig

        for n, warp in enumerate(warps_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp resized")
            output_dict[f'warp_resized_{n}'] = fig

    certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)
    H, W = certainties_max.shape

    warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]

    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = np.clip(imA * 255, 0, 255).astype(np.uint8)

    return certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all_resized, warps_all_resized


def extract_keypoints_and_colors_v2(imA, imB_compound, certainties_max, certainties_max_idcs, matches, roma_model,
                                 verbose=False, output_dict={}):
    """Extract normalized (x, y) keypoints, projection-matrix indices, and colors from matched source and target images. RoMaV2 pixel conversion takes (H_A, W_A, H_B, W_B); its (x, y) output needs no axis swap. Return (kptsA_np, kptsB_np, kptsB_proj_matrices_idx, kptsA_color, kptsB_color)."""
    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    kptsA, kptsB = roma_model.to_pixel_coordinates(
        matches, H_A, W_A, H, W
    )

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()

    if verbose:
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imA)
        ax.set_title("Reference image, imA")
        output_dict[f'reference_image'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imB_compound[0])
        ax.set_title("Image to compare to image, imB_compound")
        output_dict[f'imB_compound'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imA))
        cax = ax.scatter(kptsA_np[:, 0], H_A - kptsA_np[:, 1], s=.03)
        ax.set_title("Keypoints in imA")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsA'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_compound[0]))
        cax = ax.scatter(kptsB_np[:, 0], H - kptsB_np[:, 1], s=.03)
        ax.set_title("Keypoints in imB")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        output_dict[f'kptsB'] = fig

    kptsA_col = np.round(kptsA_np[:, 0]).astype(int)  # x = col
    kptsA_row = np.round(kptsA_np[:, 1]).astype(int)  # y = row
    kptsA_color = imA[np.clip(kptsA_row, 0, H_A - 1), np.clip(kptsA_col, 0, W_A - 1)]

    target_shape = imB_compound[0].shape[:2]  # (H_B, W_B)
    imB_compound = [cv2.resize(img, (target_shape[1], target_shape[0])) for img in imB_compound]
    imB_compound_np = np.stack(imB_compound, axis=0)
    H_B, W_B, _ = imB_compound[0].shape

    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()
    kptsA_in_cert_row = np.clip(np.round(kptsA_np[:, 1]).astype(int), 0, H - 1)  # y -> row
    kptsA_in_cert_col = np.clip(np.round(kptsA_np[:, 0]).astype(int), 0, W - 1)  # x -> col
    kptsB_proj_matrices_idx = certainties_max_idcs_np[kptsA_in_cert_row, kptsA_in_cert_col]

    kptsB_col = np.round(kptsB_np[:, 0]).astype(int)  # x = col
    kptsB_row = np.round(kptsB_np[:, 1]).astype(int)  # y = row
    if H != H_B or W != W_B:
        kptsB_in_imB_col = np.clip(np.round(kptsB_np[:, 0] / W * W_B).astype(int), 0, W_B - 1)
        kptsB_in_imB_row = np.clip(np.round(kptsB_np[:, 1] / H * H_B).astype(int), 0, H_B - 1)
    else:
        kptsB_in_imB_col = np.clip(kptsB_col, 0, W_B - 1)
        kptsB_in_imB_row = np.clip(kptsB_row, 0, H_B - 1)
    kptsB_color = imB_compound_np[kptsB_proj_matrices_idx, kptsB_in_imB_row, kptsB_in_imB_col]

    if verbose:
        imB_np = imB_compound_np[
            certainties_max_idcs_np,
            np.arange(H).reshape(-1, 1),
            np.arange(W)
        ]
        print("imB_np.shape:", imB_np.shape)
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_np))
        cax = ax.scatter(kptsB_np[:, 0], H - kptsB_np[:, 1], s=.03)
        ax.set_title("np.flipud(imB_np)")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        output_dict[f'imB_np_flipud'] = fig

    kptsA_np_norm = np.zeros_like(kptsA_np)
    kptsA_np_norm[:, 0] = kptsA_np[:, 0] / W * 2.0 - 1.0  # x / W
    kptsA_np_norm[:, 1] = kptsA_np[:, 1] / H * 2.0 - 1.0  # y / H

    kptsB_np_norm = np.zeros_like(kptsB_np)
    kptsB_np_norm[:, 0] = kptsB_np[:, 0] / W * 2.0 - 1.0  # x / W
    kptsB_np_norm[:, 1] = kptsB_np[:, 1] / H * 2.0 - 1.0  # y / H

    return kptsA_np_norm, kptsB_np_norm, kptsB_proj_matrices_idx, kptsA_color, kptsB_color


def prepare_tensor(input_array, device):
    """
    Converts an input array to a torch tensor, clones it, and detaches it for safe computation.
    Args:
        input_array (array-like): The input array to convert.
        device (str or torch.device): The device to move the tensor to.
    Returns:
        torch.Tensor: A detached tensor clone of the input array on the specified device.
    """
    if not isinstance(input_array, torch.Tensor):
        return torch.tensor(input_array, dtype=torch.float32).to(device).clone().detach()
    return input_array.clone().detach().to(device).to(torch.float32)


def triangulate_points(P1, P2, k1_x, k1_y, k2_x, k2_y, device="cuda"):
    """Triangulate a batch of homogeneous 3D points. P1 and P2 have shape (batch_size, 4, 4) or (4, 4); k1_x, k1_y, k2_x, and k2_y contain NDC coordinates of shape (batch_size,). Return X with shape (batch_size, 4)."""
    EPS = 1e-4
    # Ensure inputs are tensors

    P1 = prepare_tensor(P1, device)
    P2 = prepare_tensor(P2, device)
    k1_x = prepare_tensor(k1_x, device)
    k1_y = prepare_tensor(k1_y, device)
    k2_x = prepare_tensor(k2_x, device)
    k2_y =  prepare_tensor(k2_y, device)
    batch_size = k1_x.shape[0]

    # Expand P1 and P2 if they are not batched
    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract columns from P1 and P2
    P1_0 = P1[:, :, 0]  # Shape: (batch_size, 4)
    P1_1 = P1[:, :, 1]
    P1_2 = P1[:, :, 2]

    P2_0 = P2[:, :, 0]
    P2_1 = P2[:, :, 1]
    P2_2 = P2[:, :, 2]

    # Reshape kx and ky to (batch_size, 1)
    k1_x = k1_x.view(-1, 1)
    k1_y = k1_y.view(-1, 1)
    k2_x = k2_x.view(-1, 1)
    k2_y = k2_y.view(-1, 1)

    # Construct the equations for each batch
    # For camera 1
    A1 = P1_0 - k1_x * P1_2  # Shape: (batch_size, 4)
    A2 = P1_1 - k1_y * P1_2
    # For camera 2
    A3 = P2_0 - k2_x * P2_2
    A4 = P2_1 - k2_y * P2_2

    # Stack the equations
    A = torch.stack([A1, A2, A3, A4], dim=1)  # Shape: (batch_size, 4, 4)

    # Right-hand side (constants)
    b = -A[:, :, 3]  # Shape: (batch_size, 4)
    A_reduced = A[:, :, :3]  # Coefficients of x, y, z

    # Solve using torch.linalg.lstsq (supports batching)
    X_xyz = torch.linalg.lstsq(A_reduced, b.unsqueeze(2)).solution.squeeze(2)  # Shape: (batch_size, 3)

    # Append 1 to get homogeneous coordinates
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=X_xyz.device)
    X = torch.cat([X_xyz, ones], dim=1)  # Shape: (batch_size, 4)

    # Now compute the errors of projections.
    seeked_splats_proj1 = (X.unsqueeze(1) @ P1).squeeze(1)
    seeked_splats_proj1 = seeked_splats_proj1 / (EPS + seeked_splats_proj1[:, [3]])
    seeked_splats_proj2 = (X.unsqueeze(1) @ P2).squeeze(1)
    seeked_splats_proj2 = seeked_splats_proj2 / (EPS + seeked_splats_proj2[:, [3]])
    proj1_target = torch.concat([k1_x, k1_y], dim=1)
    proj2_target = torch.concat([k2_x, k2_y], dim=1)
    errors_proj1 = torch.abs(seeked_splats_proj1[:, :2] - proj1_target).sum(1).detach().cpu().numpy()
    errors_proj2 = torch.abs(seeked_splats_proj2[:, :2] - proj2_target).sum(1).detach().cpu().numpy()

    return X, errors_proj1, errors_proj2


def select_best_keypoints(
        NNs_triangulated_points, NNs_errors_proj1, NNs_errors_proj2, device="cuda"):
    """Select triangulated points with the lowest reprojection errors across neighbors. Points have shape (num_nns, num_points, dim); each error array has shape (num_nns, num_points). Return selected points, minimum errors, and neighbor indices identifying the camera pairs."""

    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)

    selected_nn_indices = np.argmin(NNs_errors_proj, axis=0)
    indices = torch.from_numpy(selected_nn_indices).long().to(device)

    # Create index tensor for the second dimension
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)

    # Use advanced indexing to select elements
    NNs_triangulated_points_selected = NNs_triangulated_points[indices, n_indices, :]  # Shape: [N, k]

    return NNs_triangulated_points_selected, np.min(NNs_errors_proj, axis=0), selected_nn_indices


def compute_triangulation_angle(xyz, cam1_centers, cam2_centers):
    """Compute angles in degrees between rays from each 3D point to its two camera centers. All inputs have shape (N, 3); return an (N,) NumPy array."""
    ray1 = xyz - cam1_centers  # (N, 3)
    ray2 = xyz - cam2_centers  # (N, 3)

    ray1_norm = ray1 / (np.linalg.norm(ray1, axis=1, keepdims=True) + 1e-8)
    ray2_norm = ray2 / (np.linalg.norm(ray2, axis=1, keepdims=True) + 1e-8)

    cos_angle = np.sum(ray1_norm * ray2_norm, axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))

    return angles


def filter_triangulated_points(
    xyz, colors, proj_errors,
    ref_cam_center,
    cam1_centers=None,
    cam2_centers=None,
    max_reproj_error=0.02,
    min_depth=0.1,
    max_depth=1000.0,
    min_triangulation_angle=1.0,
    use_statistical_filter=True,
    nb_neighbors=20,
    std_ratio=2.0,
    use_radius_filter=False,
    radius=0.5,
    min_neighbors=5,
    verbose=True
):
    """Filter triangulated points by reprojection error, depth, triangulation angle, and optional statistical or radius outlier removal. xyz and colors have shape (N, 3); proj_errors has shape (N,). Camera centers support angle filtering in degrees. nb_neighbors/std_ratio configure SOR; radius/min_neighbors configure radius filtering. Return filtered coordinates, colors, and the input-aligned validity mask."""
    n_total = xyz.shape[0]
    valid_mask = np.ones(n_total, dtype=bool)

    reproj_mask = proj_errors <= max_reproj_error
    valid_mask &= reproj_mask
    if verbose:
        print(f"  [Filter] Reprojection error (<{max_reproj_error}): {reproj_mask.sum()}/{n_total} points kept")

    if isinstance(ref_cam_center, torch.Tensor):
        ref_cam_center = ref_cam_center.cpu().numpy()
    depths = np.linalg.norm(xyz - ref_cam_center.reshape(1, 3), axis=1)
    depth_mask = (depths >= min_depth) & (depths <= max_depth)
    valid_mask &= depth_mask
    if verbose:
        print(f"  [Filter] Depth range ({min_depth}-{max_depth}): {(valid_mask).sum()}/{n_total} points kept")

    if cam1_centers is not None and cam2_centers is not None and min_triangulation_angle > 0:
        tri_angles = compute_triangulation_angle(xyz, cam1_centers, cam2_centers)
        angle_mask = tri_angles >= min_triangulation_angle
        valid_mask &= angle_mask
        if verbose:
            print(f"  [Filter] Triangulation angle (>={min_triangulation_angle}°): {valid_mask.sum()}/{n_total} points kept")
            print(f"           Angle stats: min={tri_angles[valid_mask].min():.2f}°, max={tri_angles[valid_mask].max():.2f}°, mean={tri_angles[valid_mask].mean():.2f}°")

    if use_statistical_filter and valid_mask.sum() > nb_neighbors:
        try:
            import open3d as o3d
            temp_xyz = xyz[valid_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(temp_xyz)

            _, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )

            temp_indices = np.where(valid_mask)[0]
            sor_mask = np.zeros(n_total, dtype=bool)
            sor_mask[temp_indices[inlier_indices]] = True
            valid_mask &= sor_mask

            if verbose:
                print(f"  [Filter] Statistical Outlier Removal (k={nb_neighbors}, std={std_ratio}): {valid_mask.sum()}/{n_total} points kept")
        except ImportError:
            if verbose:
                print("  [Filter] Warning: open3d not installed, skipping statistical outlier removal")

    if use_radius_filter and valid_mask.sum() > min_neighbors:
        try:
            import open3d as o3d
            temp_xyz = xyz[valid_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(temp_xyz)

            _, inlier_indices = pcd.remove_radius_outlier(
                nb_points=min_neighbors,
                radius=radius
            )

            temp_indices = np.where(valid_mask)[0]
            ror_mask = np.zeros(n_total, dtype=bool)
            ror_mask[temp_indices[inlier_indices]] = True
            valid_mask &= ror_mask

            if verbose:
                print(f"  [Filter] Radius Outlier Removal (r={radius}, min_nb={min_neighbors}): {valid_mask.sum()}/{n_total} points kept")
        except ImportError:
            if verbose:
                print("  [Filter] Warning: open3d not installed, skipping radius outlier removal")

    return xyz[valid_mask], colors[valid_mask], valid_mask


def save_roma_pointcloud_ply(xyz, colors, source_path, filename="dense_v2.ply"):
    """Save xyz (N, 3) and RGB colors (0-255) to filename under source_path."""
    save_dir = os.path.join(source_path, "sparse", "0")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    colors = np.clip(colors, 0, 255).astype(np.uint8)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = 0.0
    elements['ny'] = 0.0
    elements['nz'] = 0.0
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(save_path)

    print(f"[RoMa V2 Init] Saved {xyz.shape[0]} points to {save_path}")


def create_roma_v2_model(opt, device="cuda"):
    """Create a RoMaV2 model on device using opt.roma_model settings."""

    if hasattr(opt, 'roma_model'):
        if opt.roma_model == "indoors":
            setting = "scannet1500"
        elif opt.roma_model == "outdoors":
            setting = "fast"
        else:
            setting = "fast"
    else:
        setting = "fast"

    print(f"[RoMa V2] Creating model with setting: {setting}")

    cfg = RoMaV2.Cfg(setting=setting, compile=True)

    roma_model = RoMaV2(cfg)

    return roma_model


def init_gaussians_with_corr(
    gaussians,
    scene,
    opt,
    device,
    verbose=True,
    roma_model=None,
    pair_file=None,
    use_unfiltered_init=False,
    init_scale_factor=1.0,
    init_opacity=0.1,
):
    """Initialize scene Gaussians from RoMaV2 matches. Accept an optional preloaded roma_model and pair_file for precomputed neighbors. use_unfiltered_init selects pre-filter points for in-memory initialization. Return the view stack, selected neighbor indices, and visualizations."""
    if roma_model is None:
        roma_model = create_roma_v2_model(opt, device)

    M = opt.matches_per_ref
    upper_thresh = 0.9
    min_certainty_thresh = getattr(opt, 'roma_certainty_thresh', 0.5)
    expansion_factor = 1
    keypoint_fit_error_tolerance = opt.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()

    roma_mask_dir = getattr(opt, 'roma_mask_dir', '')
    cam_masks = {}
    if roma_mask_dir and os.path.isdir(roma_mask_dir):
        loaded = 0
        for cam in viewpoint_stack:
            mask_path = os.path.join(roma_mask_dir, f"{cam.image_name}.png")
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
                cam_masks[cam.uid] = m
                loaded += 1
        print(f"[RoMa Mask] Loaded {loaded}/{len(viewpoint_stack)} masks from {roma_mask_dir}")
    NUM_REFERENCE_FRAMES = min(opt.num_refs, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = min(opt.nns_per_ref , len(viewpoint_stack))

    if pair_file is not None and os.path.exists(pair_file):
        print(f"[Init] Using precomputed pair file: {pair_file}")
        pair_dict, num_images_in_pair = read_pair_file(pair_file, num_src_views=NUM_NNS_PER_REFERENCE)

        num_cameras = len(viewpoint_stack)
        print(f"[Pair File] Number of training cameras: {num_cameras}, pair.txt has {num_images_in_pair} images")

        colmap_id_to_vp_idx = {cam.colmap_id: idx for idx, cam in enumerate(viewpoint_stack)}

        pair_dict_vp = {}
        for ref_colmap_id, src_colmap_ids in pair_dict.items():
            if ref_colmap_id in colmap_id_to_vp_idx:
                vp_ref_idx = colmap_id_to_vp_idx[ref_colmap_id]
                vp_src_list = []
                for src_colmap_id in src_colmap_ids:
                    if src_colmap_id in colmap_id_to_vp_idx:
                        vp_src_list.append(colmap_id_to_vp_idx[src_colmap_id])
                if vp_src_list:
                    pair_dict_vp[vp_ref_idx] = vp_src_list

        print(f"[Pair File] Converted {len(pair_dict_vp)} camera pairs to viewpoint indices")

        if len(pair_dict_vp) == 0:
            # Common cause: pair.txt's ref ids don't match cam.colmap_id.
            # This happens when selectpairs.py wrote 1..N indices but the COLMAP
            # reconstruction has image_ids that don't start from 1 (e.g. 1211..1655),
            # or when --eval filtered out every candidate src for every ref.
            pair_ids_sample = list(pair_dict.keys())[:5]
            vp_ids_sample = list(colmap_id_to_vp_idx.keys())[:5]
            raise RuntimeError(
                "[Pair File] 0 camera pairs matched viewpoint indices. "
                "pair.txt ref ids and cam.colmap_id are in different id spaces. "
                f"pair.txt ref ids sample: {pair_ids_sample}, "
                f"cam.colmap_id sample: {vp_ids_sample}. "
                "If your COLMAP image_ids don't start at 1, regenerate pair.txt "
                "with the fixed selectpairs.py (which writes real image_ids)."
            )

        valid_camera_indices = list(pair_dict_vp.keys())
        if NUM_REFERENCE_FRAMES < len(valid_camera_indices):
            step = len(valid_camera_indices) // NUM_REFERENCE_FRAMES
            selected_indices = sorted([valid_camera_indices[i * step] for i in range(NUM_REFERENCE_FRAMES)])
        else:
            selected_indices = sorted(valid_camera_indices)

        closest_indices_selected = np.zeros((num_cameras, NUM_NNS_PER_REFERENCE), dtype=np.int64)
        for vp_idx in range(num_cameras):
            if vp_idx in pair_dict_vp:
                src_list = pair_dict_vp[vp_idx]
                for i in range(NUM_NNS_PER_REFERENCE):
                    if i < len(src_list):
                        closest_indices_selected[vp_idx, i] = src_list[i]
                    elif src_list:
                        closest_indices_selected[vp_idx, i] = src_list[-1]
                    else:
                        closest_indices_selected[vp_idx, i] = vp_idx
            else:
                closest_indices_selected[vp_idx, :] = vp_idx

        if verbose:
            print(f"[Pair File] Selected {len(selected_indices)} reference frames")
            print(f"[Pair File] Sample closest indices for camera 0: {closest_indices_selected[0]}")
            if 0 in pair_dict_vp:
                print(f"[Pair File] Camera 0 colmap_id: {viewpoint_stack[0].colmap_id}, name: {viewpoint_stack[0].image_name}")
                print(f"[Pair File] Camera 0 src colmap_ids: {[viewpoint_stack[s].colmap_id for s in pair_dict_vp[0][:3]]}")
    else:
        print("[Init] Using K-means for camera selection")
        viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

        selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
        selected_indices = sorted(selected_indices)

        # Find the k-closest vectors for each vector
        viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
        closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
        if verbose: print("Indices of k-closest vectors for each vector:\n", closest_indices)

        closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_colors = []
    all_new_proj_errors = []
    all_new_cam_centers = []
    all_new_cam1_centers = []
    all_new_cam2_centers = []

    # Run roma_model.match once to initialize the model (warm up)
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        preds = roma_model.match(imA, imB)
        print("Once run full roma_model.match warp_AB.shape:", preds["warp_AB"].shape)
        print("Once run full roma_model.match overlap_AB.shape:", preds["overlap_AB"].shape)
        del preds
        torch.cuda.empty_cache()

    for source_idx in tqdm(sorted(selected_indices)):
        # 1. Compute keypoints and warping for all the neighboring views
        with torch.no_grad():
            # Call the aggregation function to get imA and imB_compound
            certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all = aggregate_confidences_and_warps_v2(
                viewpoint_stack=viewpoint_stack,
                closest_indices=closest_indices_selected,
                roma_model=roma_model,
                source_idx=source_idx,
                verbose=verbose, output_dict=visualizations
            )


        # Triangulate keypoints
        with torch.no_grad():
            matches = warps_max
            certainty = certainties_max
            certainty = certainty.clone()

            ref_cam = viewpoint_stack[source_idx]
            keep_masks_np = []
            mask_sources = []
            if ref_cam.uid in cam_masks:
                keep_masks_np.append(cam_masks[ref_cam.uid])
                mask_sources.append("roma_mask")
            alpha_mask_np = get_camera_alpha_mask_np(ref_cam)
            if alpha_mask_np is not None and (alpha_mask_np < 0.5).any():
                keep_masks_np.append(alpha_mask_np)
                mask_sources.append("alpha")

            if keep_masks_np:
                H_cert, W_cert = certainty.shape
                mask_resized = torch.ones_like(certainty)
                for keep_mask_np in keep_masks_np:
                    mask_np = (keep_mask_np >= 0.5).astype(np.float32)
                    H_mask, W_mask = mask_np.shape
                    if H_cert != H_mask or W_cert != W_mask:
                        mask_tensor = torch.from_numpy(
                            cv2.resize(mask_np, (W_cert, H_cert), interpolation=cv2.INTER_NEAREST)
                        ).to(certainty.device)
                    else:
                        mask_tensor = torch.from_numpy(mask_np).to(certainty.device)
                    mask_resized *= mask_tensor
                n_filtered = (mask_resized < 0.5).sum().item()
                certainty *= mask_resized
                if source_idx == sorted(selected_indices)[0]:
                    print(
                        f"[RoMa Mask] Filtering {n_filtered}/{certainty.numel()} pixels "
                        f"from certainty map ({'+'.join(mask_sources)})"
                    )

            certainty[certainty < min_certainty_thresh] = 0
            certainty[certainty > upper_thresh] = 1

            matches, certainty = (
                matches.reshape(-1, 4),
                certainty.reshape(-1),
            )

            valid_mask = certainty > 0
            valid_count = valid_mask.sum().item()
            if valid_count < 100:
                print(f"  [Warning] Camera {source_idx}: only {valid_count} points above certainty threshold {min_certainty_thresh}, skipping...")
                continue

            MULTINOMIAL_MAX = (1 << 24)
            n_samples = min(expansion_factor * M, valid_count)
            if certainty.numel() > MULTINOMIAL_MAX or valid_count < certainty.numel() // 4:
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                if valid_indices.numel() > MULTINOMIAL_MAX:
                    perm = torch.randperm(valid_indices.numel(), device=certainty.device)[:MULTINOMIAL_MAX]
                    valid_indices = valid_indices[perm]
                weights = certainty[valid_indices]
                local_idx = torch.multinomial(weights, num_samples=n_samples, replacement=False)
                good_samples = valid_indices[local_idx]
            else:
                good_samples = torch.multinomial(certainty, num_samples=n_samples, replacement=False)

        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all
        reference_image_dict = {
            "ref_image": imA,
            "NNs_images": imB_compound,
            "certainties_all": certainties_all,
            "warps_all": warps_all,
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []

        }
        with torch.no_grad():
            for NN_idx in range(len(warps_all)):
                matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]

                # Extract keypoints and colors
                kptsA_np, kptsB_np, kptsB_proj_matrices_idcs, kptsA_color, kptsB_color = extract_keypoints_and_colors_v2(
                    imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
                )

                proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
                proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, NN_idx]].full_proj_transform
                triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
                    P1=torch.stack([proj_matrices_A] * M, axis=0),
                    P2=torch.stack([proj_matrices_B] * M, axis=0),
                    k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
                    k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

                reference_image_dict["triangulated_points"].append(triangulated_points)
                reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
                reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)

        with torch.no_grad():
            NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors, selected_nn_indices = select_best_keypoints(
                NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
                NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
                NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))

        # 4. Save as gaussians
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        with torch.no_grad():
            new_xyz = NNs_triangulated_points_selected[:, :-1]
            all_new_xyz.append(new_xyz)  # seeked_splats
            all_new_proj_errors.append(NNs_triangulated_points_selected_proj_errors)
            all_new_cam_centers.append(viewpoint_cam1.camera_center.clone().detach())

            cam1_center = viewpoint_cam1.camera_center.clone().detach().cpu().numpy()
            cam1_centers_for_points = np.tile(cam1_center.reshape(1, 3), (N, 1))  # (N, 3)
            all_new_cam1_centers.append(cam1_centers_for_points)

            cam2_centers_for_points = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                nn_idx = selected_nn_indices[i]
                cam2_idx = closest_indices_selected[source_idx, nn_idx]
                cam2_centers_for_points[i] = viewpoint_stack[cam2_idx].camera_center.clone().detach().cpu().numpy()
            all_new_cam2_centers.append(cam2_centers_for_points)

            all_new_colors.append(kptsA_color)

    all_new_xyz = torch.cat(all_new_xyz, dim=0)
    all_new_proj_errors = np.concatenate(all_new_proj_errors, axis=0)
    all_new_colors_np = np.concatenate(all_new_colors, axis=0)
    all_new_cam1_centers_np = np.concatenate(all_new_cam1_centers, axis=0)  # (N, 3)
    all_new_cam2_centers_np = np.concatenate(all_new_cam2_centers, axis=0)  # (N, 3)

    scene_center = torch.stack(all_new_cam_centers, dim=0).mean(dim=0)

    all_new_xyz_np = all_new_xyz.cpu().numpy()

    print(f"\n[RoMa Init] Before filtering: {all_new_xyz.shape[0]} points")

    save_roma_pointcloud_ply(
        xyz=all_new_xyz_np,
        colors=all_new_colors_np,
        source_path=scene.source_path,
        filename="dense_v2_unfiltered.ply"
    )

    min_depth = getattr(opt, 'filter_min_depth', 0.1)
    max_depth = getattr(opt, 'filter_max_depth', 500.0)
    min_tri_angle = getattr(opt, 'filter_min_tri_angle', 1.0)
    sor_k = getattr(opt, 'filter_sor_k', 30)
    sor_std = getattr(opt, 'filter_sor_std', 2.0)
    use_sor = getattr(opt, 'filter_use_sor', True)

    filtered_xyz, filtered_colors, valid_mask = filter_triangulated_points(
        xyz=all_new_xyz_np,
        colors=all_new_colors_np,
        proj_errors=all_new_proj_errors,
        ref_cam_center=scene_center,
        cam1_centers=all_new_cam1_centers_np,
        cam2_centers=all_new_cam2_centers_np,
        max_reproj_error=keypoint_fit_error_tolerance,
        min_depth=min_depth,
        max_depth=max_depth,
        min_triangulation_angle=min_tri_angle,
        use_statistical_filter=use_sor,
        nb_neighbors=sor_k,
        std_ratio=sor_std,
        verbose=True
    )

    print(f"[RoMa Init] After filtering: {valid_mask.sum()} points ({100*valid_mask.sum()/len(valid_mask):.1f}%)\n")

    init_mask_np = np.ones_like(valid_mask, dtype=bool) if use_unfiltered_init else valid_mask
    init_mask = torch.from_numpy(init_mask_np)
    init_kind = "unfiltered" if use_unfiltered_init else "filtered"
    print(f"[RoMa Init] Using {init_kind} dense init in memory: {init_mask_np.sum()} points")

    parameters = gaussians.point_parameters(
        all_new_xyz[init_mask].to(device),
        all_new_colors_np[init_mask_np].astype(np.float32) / 255.0,
        init_scale_factor, init_opacity,
    )
    gaussians.densification_postfix(*parameters)

    save_roma_pointcloud_ply(
        xyz=filtered_xyz,
        colors=filtered_colors,
        source_path=scene.source_path,
        filename="dense_v2.ply"
    )

    return viewpoint_stack, closest_indices_selected, visualizations


