# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.mesh import get_urdf_mesh_collection
from lygra.kinematics import build_kinematics_tree
from lygra.utils.geom_utils import get_tangent_plane
from pathlib import Path
import importlib.util
import torch 
import torch.nn.functional as F 
import time 
import trimesh
import numpy as np
import math 
import warnings


def farthest_point_sampling(vectors, k, pos_weight=1.0, dir_weight=1.0):
    """
    Subsample k vectors using farthest point sampling considering both position and direction.
    
    Args:
        vectors: [N, 6] array where [:, :3] are positions, [:, 3:] are directions
        k: number of samples to select
        pos_weight: weight for position diversity
        dir_weight: weight for direction diversity
    """
    n = vectors.shape[0]
    positions = vectors[:, :3]
    directions = vectors[:, 3:]
    
    # Normalize directions to unit length for meaningful distance computation
    directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    selected_indices = []
    remaining_indices = set(range(n))
    
    # Start with a random point
    first_idx = np.random.randint(n)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    while len(selected_indices) < k:
        # Compute combined distance from each remaining point to all selected points
        max_min_distances = np.zeros(len(remaining_indices))
        
        for i, idx in enumerate(remaining_indices):
            # Position distances
            pos_dist = np.linalg.norm(positions[selected_indices] - positions[idx], axis=1)

            dir_similarity = np.abs(np.sum(directions_norm[selected_indices] * directions_norm[idx], axis=1))
            dir_dist = 1.0 - dir_similarity

            combined_dist = pos_weight * pos_dist + dir_weight * dir_dist
            max_min_distances[i] = np.min(combined_dist)

        best_remaining_idx = np.argmax(max_min_distances)
        best_idx = list(remaining_indices)[best_remaining_idx]
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return np.array(selected_indices)



def upsample_to(arr, n):
    if len(arr) >= n:
        #return arr[farthest_point_sampling(arr, n)]
        perm = np.random.permutation(len(arr))
        return arr[perm[:n]]
    else:
        indices = np.random.choice(len(arr), size=n, replace=True)
        return arr[indices]


def generate_mesh_group(
    mesh, 
    n=3000, 
    n_keyvector=6, 
    patch_pos_dist_tol=0.0075, 
    patch_rot_dist_tol=3.1415926 * 0.25
):
    # print(patch_pos_dist_tol, patch_rot_dist_tol)
    # we first sample point and normal.
    points, face_indices = trimesh.sample.sample_surface(mesh, count=n)
    normals = mesh.face_normals[face_indices]

    perm = np.random.permutation(n)
    points = points[perm]
    normals = normals[perm]

    all_centroids = []
    all_items = []
    all_angle_ranges = []
    all_ids = []
    all_keyvectors = []

    group_id = 0
    while len(points) > 0:
        # next_idx = np.random.randint(0, len(points), 1)[0]
        point = points[0:1]
        normal = normals[0:1]
        centroid = np.concatenate([point, normal], axis=-1)

        all_items.append(centroid)
        all_ids.append(group_id)
        all_centroids.append(centroid)

        points = points[1:]
        normals = normals[1:]

        dist = np.linalg.norm(points - point, axis=-1).reshape(-1)
        rot = np.arccos(np.clip((normal * normals).sum(axis=-1), -1 + 1e-9, 1 - 1e-9)).reshape(-1)
        mask1 = dist < patch_pos_dist_tol
        mask2 = rot < patch_rot_dist_tol
        mask = np.logical_and(mask1, mask2)

        selection = np.where(mask)
        remain = np.where(~mask)

        group_angle_range = np.arccos(np.clip((normal * normals[selection]).sum(axis=-1), -1 + 1e-9, 1 - 1e-9)).reshape(-1)
        if len(group_angle_range) == 0:
            group_angle_range = 0.2
        else:
            group_angle_range = max([np.max(np.abs(group_angle_range)), 0.2])

        new_items = np.concatenate([points[selection], normals[selection]], axis=-1)
        all_items.append(new_items)
        all_angle_ranges.append(group_angle_range)
        all_keyvectors.append(
            upsample_to(np.concatenate([centroid, new_items]), n_keyvector)
        )

        for i in range(len(selection[0])):
            all_ids.append(group_id)

        points = points[remain]
        normals = normals[remain]
        group_id += 1

    all_ids = np.array(all_ids)
    all_items = np.concatenate(all_items)
    all_angle_ranges = np.array(all_angle_ranges)

    return {
        "ids": all_ids, 
        "all": all_items, 
        "group_angle_range": all_angle_ranges,
        "group_vectors": all_keyvectors,    # list of [n_keyvector, 6]
        "group_centroid": all_centroids     # list of [1, 6] vector.
    }



def get_robot_link_mesh_group(
    urdf_path, 
    mesh_scale=1.0,
    patch_pos_dist_tol=0.0075, 
    patch_rot_dist_tol=3.1415926 * 0.25
):
    tree = build_kinematics_tree(urdf_path)
    mesh_collection = get_urdf_mesh_collection(urdf_path, tree, mesh_scale=mesh_scale)

    out = {}
    for k, v in mesh_collection.items():
        if v.vertices.shape[0] == 0:
            continue

        out[k] = generate_mesh_group(
            mesh=v, 
            patch_pos_dist_tol=patch_pos_dist_tol, 
            patch_rot_dist_tol=patch_rot_dist_tol
        )
    return out


def get_point_self_transformation_matrix(pos, normal):
    """
    Get self transformation matrix.
        
    Args:
        pos:    [N, 3]
        normal: [N, 3]

    Returns:
        matrix: [N, N].  M[i][j] is position of point j as viewed from i's frame.

    """
    normal_x, normal_y = get_tangent_plane(normal)
    rotation = torch.stack([normal_x, normal_y, normal], dim=2) # [N, 3, 3]
    translation = pos
    
    X = pos.unsqueeze(0).expand(pos.size(0), -1, -1)    # [N, N, 3]
    X = X - pos.unsqueeze(1)   # [N, N, 3]

    # we will have X = [N, N, 3] , X[i][j] is position of point j as viewed from i's frame.
    X = torch.bmm(X, rotation)
    return X


def get_support_point_mask(pos, normal, radius_lists=[], z_threshold=0.002, n_violation_threshold=5):
    """
        Get the mask of "support point".
        
        Args:
        pos:    [N, 3]
        normal: [N, 3]

        Returns:
        masks.  a list of masks. shape = [N,]

    """
    X = get_point_self_transformation_matrix(pos, normal)
    z = X[:, :, 2]
    r2 = (X[:, :, :2] ** 2).sum(dim=-1)
    
    mask_z_position = z > z_threshold

    all_masks = []
    for r in radius_lists:
        mask_xy_position = r2 < (r ** 2)
        mask = mask_xy_position & mask_z_position   # [N, N]
        mask = mask.sum(dim=1) > n_violation_threshold
        all_masks.append(~mask)
    return all_masks
