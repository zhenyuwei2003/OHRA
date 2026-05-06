# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def transform_point(transform, point):
    """
    Args:
        transform: [4, 4] matrix.
        point:     [N, 3] 
    """
    assert len(transform.shape) == 2
    return point @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]


def batch_apply_transform_np(point, transform):
    '''
        point: [B, 3]
        transform: [B, 4, 4]
        Returns: [B, 3]
    '''
    # Convert points to homogeneous coordinates: [B, 4]
    ones = np.ones((point.shape[0], 1))
    point_h = np.concatenate([point, ones], axis=-1)  # [B, 4]
    point_transformed_h = np.einsum('bij,bj->bi', transform, point_h)
    return point_transformed_h[:, :3]


def batch_apply_rotation_np(point, transform):
    '''
        point: [B, 3]
        transform: [B, 4, 4]
        Returns: [B, 3]
    '''
    rotation = transform[:, :3, :3]
    rotated_point = np.einsum('bij,bj->bi', rotation, point)
    return rotated_point


def batch_axis_angle(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Fast conversion from batched axis-angle to [B, 4, 4] rotation matrices.

    Args:
        angle: [B] rotation angles in radians
        axis: [B, 3] rotation axes (not necessarily normalized)

    Returns:
        [B, 4, 4] batched homogeneous rotation matrices
    """
    B = angle.shape[0]
    device = angle.device
    dtype = angle.dtype

    # Normalize axis
    axis = torch.nn.functional.normalize(axis, dim=1, eps=1e-8)  # [B, 3]
    x, y, z = axis.unbind(dim=1)  # Each is [B]

    cos = angle.cos()
    sin = angle.sin()
    one_minus_cos = 1 - cos

    # Compute outer products of axis
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z

    xs = x * sin
    ys = y * sin
    zs = z * sin

    # Build rotation matrix directly [B, 3, 3]
    rot = torch.stack([
        cos + xx * one_minus_cos,     xy * one_minus_cos - zs,     xz * one_minus_cos + ys,
        xy * one_minus_cos + zs,      cos + yy * one_minus_cos,    yz * one_minus_cos - xs,
        xz * one_minus_cos - ys,      yz * one_minus_cos + xs,     cos + zz * one_minus_cos
    ], dim=-1).reshape(B, 3, 3)

    # Create homogeneous matrix [B, 4, 4]
    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = rot
    return T


def batch_translation(delta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Compute batched 4x4 homogeneous translation matrices.

    Args:
        delta: [B] translation magnitudes
        axis: [B, 3] translation directions (not necessarily normalized)

    Returns:
        [B, 4, 4] translation matrices
    """
    B = delta.shape[0]
    device = delta.device
    dtype = delta.dtype

    # Normalize axis to get direction
    direction = torch.nn.functional.normalize(axis, dim=1, eps=1e-8)  # [B, 3]
    
    # Translation vector = delta * direction
    t = delta.unsqueeze(1) * direction  # [B, 3]

    # Initialize identity matrices
    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, 3] = t  # Insert translation vector
    return T


def batch_object_transform(object_pose, object_pos, object_normal=None):
    """
    Args:
        object_pos:     [N, 3]
        object_normal:  [N, 3]
        object_poses:   [B, 4, 4]

    Returns:
        pos:            [B, N, 3]
        normal:         [B, N, 3] (optional)
    """
    B = object_pose.shape[0]

    # We first apply the transformation on these points!
    object_pos = object_pos[None].repeat(B, 1, 1)               # [B, N, 3]
    if object_normal is not None:
        object_normal = object_normal[None].repeat(B, 1, 1)     # [B, N, 3]

    object_pos = torch.bmm(object_pos, object_pose[:, :3, :3].permute(0, 2, 1)) + object_pose[:, :3, 3].unsqueeze(1)    # [B, N, 3]
    if object_normal is not None:
        object_normal = torch.bmm(object_normal, object_pose[:, :3, :3].permute(0, 2, 1))                               # [B, N, 3]

    return {"pos": object_pos, "normal": object_normal}
