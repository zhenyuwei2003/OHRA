# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
from lygra import gem
from lygra.kinematics import batch_fk
from lygra.utils.transform_utils import batch_object_transform


def batch_object_hand_collision_check(
    tree,
    object_point,
    q=None,
    target_links=None,
    hand_mesh_buffer=None,
    mesh=None,
    ret_collision=False
):  
    """
    Subroutine to detect hand object collision.

    Returns:
        mask:   [batch_size, ] torch.Tensor (bool), 1 if no collision.
    """
    if hand_mesh_buffer:
        assert False # TODO()
        v_tensor = hand_mesh_buffer["v"]
        f_tensor = hand_mesh_buffer["f"]
        n_tensor = hand_mesh_buffer["n"]
        v_idx_tensor = hand_mesh_buffer["vi"]
        f_idx_tensor = hand_mesh_buffer["fi"]

    else:
        v_tensor = torch.from_numpy(mesh['v']).cuda().float()
        f_tensor = torch.from_numpy(mesh['f']).cuda().int()
        n_tensor = torch.from_numpy(mesh['n']).cuda().float()
        v_idx_tensor = torch.from_numpy(mesh['vi']).cuda().int()
        f_idx_tensor = torch.from_numpy(mesh['fi']).cuda().int()

    if q is None:
        q = torch.from_numpy(tree.get_resting_q()).cuda() # use default. 
        q = q[None].expand(len(object_point), -1)

    mesh_pose = batch_fk(tree, q)["link"]
    mesh_pose = mesh_pose[:, mesh["body_id_to_link_id"], :]

    if target_links is None:
        batch_size = object_point.size(0)
        num_links = len(mesh["body_id_to_link_id"]) #tree.n_link()
        target_links = torch.arange(num_links)[None].repeat(batch_size, 1).to(object_point.device).int()

    out = gem.batch_point_to_link_mesh_collision(
        mesh_pose,
        target_links,
        v_tensor.reshape(-1),
        v_idx_tensor,
        f_tensor.reshape(-1),
        n_tensor.reshape(-1),
        f_idx_tensor,
        object_point
    )

    if ret_collision:
        return out.int()
    return out.logical_not().int()


def batch_hand_self_collision_check(
    tree,
    q,
    self_collision_link_pairs,
    hand_mesh_buffer=None,
    mesh=None
):  
    """
    Subroutine to detect hand object collision.

    Returns:
        mask:   [batch_size, ] torch.Tensor (bool), 1 if no collision.
    """
    if hand_mesh_buffer:
        assert False
        v_tensor = hand_mesh_buffer["v"]
        f_tensor = hand_mesh_buffer["f"]
        n_tensor = hand_mesh_buffer["n"]
        v_idx_tensor = hand_mesh_buffer["vi"]
        f_idx_tensor = hand_mesh_buffer["fi"]

    else:
        v_tensor = torch.from_numpy(mesh['v']).cuda().float()
        f_tensor = torch.from_numpy(mesh['f']).cuda().int()
        n_tensor = torch.from_numpy(mesh['n']).cuda().float()
        v_idx_tensor = torch.from_numpy(mesh['vi']).cuda().int()
        f_idx_tensor = torch.from_numpy(mesh['fi']).cuda().int()

    mesh_pose = batch_fk(tree, q)["link"]
    mesh_pose = mesh_pose[:, mesh["body_id_to_link_id"], :]

    transformed_vertex = gem.batch_link_vertex_transform(
        mesh_pose,
        v_tensor.reshape(-1, 3),
        v_idx_tensor
    )

    out = gem.batch_link_mesh_gjk(
        mesh_pose,
        v_tensor.reshape(-1, 3),
        v_idx_tensor,
        self_collision_link_pairs
    )

    out = out.bool()
    return (out.any(dim=-1)).logical_not().int(), out, transformed_vertex


def batch_filter_collision(
    tree,
    q,
    object_pose,
    object_point,
    self_collision_link_pairs,
    decomposed_mesh_data,
    ret_mask_only=True
):
    object_point = batch_object_transform(object_pose, object_point)["pos"]
    mask_object = batch_object_hand_collision_check(
        tree=tree, 
        q=q, 
        object_point=object_point, 
        mesh=decomposed_mesh_data,
        ret_collision=False
    )
    # mask_hand, out, transformed_vertex = batch_hand_self_collision_check(
    #     tree,
    #     q=q,
    #     self_collision_link_pairs=self_collision_link_pairs,
    #     mesh=decomposed_mesh_data
    # )
    # mask = mask_object & mask_hand
    mask = mask_object

    if ret_mask_only:
        return mask

    passed_idx = torch.where(mask)
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            result[k] = v[passed_idx]

    return result

