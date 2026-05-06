# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
import torch.nn.functional as F
from lygra.kinematics import batch_contact_ik, batch_fk
from lygra import gem
from tqdm import tqdm


def batch_ik(
    tree,
    contact_ids,             # [batch, n_contact]
    contact_parent_ids, 
    contact_pos_in_linkf,    # [batch, n_contact, 3]
    contact_normal_in_linkf, # [batch, n_contact, 3]
    target_contact_pos,      # [batch, n_contact, 3]
    target_contact_normal,   # [batch, n_contact, 3]
    object_pose,
    gpu_memory_pool,
    n_dof=16,
    n_retry=10,
    regularization=1e-4,
    err_thresh=0.02,
    max_iter=12,
    step_size=0.4,
    filter=True
):
    batch_size = contact_ids.shape[0]
    contact_link_ids = contact_parent_ids[contact_ids]  # [batch, n_contact]

    ik_result = batch_contact_ik(
        tree,
        target_contact_pos, 
        target_contact_normal, 
        contact_pos_in_linkf, 
        contact_normal_in_linkf, 
        contact_link_ids,   
        n_retry=n_retry, 
        step_size=step_size, 
        max_iter=max_iter, 
        regularization=regularization,
        err_thresh=err_thresh,
        alpha_contact_matching=1.0,
        all_joint_result_buffer=gpu_memory_pool.get_ik_joint_buffer(batch_size, n_retry),
        all_link_result_buffer=gpu_memory_pool.get_ik_link_buffer(batch_size, n_retry),
        jac_result_buffer=gpu_memory_pool.get_ik_jac_result_buffer(batch_size, n_retry),
        J_err_buffer=gpu_memory_pool.get_ik_jac_error_buffer(batch_size, n_retry, contact_link_ids.size(-1))
    )

    q = ik_result["q"]
    success_idx = ik_result["success"]
    q_mask = ik_result["q_mask"]

    # q:             [batch, n_dof]
    # success_idx:   [batch]
    if filter:
        result = {
            "q":                q[success_idx], 
            "q_mask":           q_mask[success_idx], 
            "object_pose":      object_pose[success_idx],
            "target_pos":       target_contact_pos[success_idx],
            "target_normal":    target_contact_normal[success_idx],
            "contact_pos":      contact_pos_in_linkf[success_idx],
            "contact_normal":   contact_normal_in_linkf[success_idx],
            "contact_link_id":  contact_link_ids[success_idx]
        }
    else:
        result = {
            "q":                q, 
            "q_mask":           q_mask, 
            "object_pose":      object_pose,
            "target_pos":       target_contact_pos,
            "target_normal":    target_contact_normal,
            "contact_pos":      contact_pos_in_linkf,
            "contact_normal":   contact_normal_in_linkf,
            "contact_link_id":  contact_link_ids
        }
    return result


def batch_contact_adjustment(
    tree,
    mesh,
    q_init,
    q_mask,
    contact_ids,
    contact_link_ids,        # [batch, n_contact]
    contact_pos_in_linkf,    # [batch, n_contact, 3]
    contact_normal_in_linkf, # [batch, n_contact, 3]
    target_contact_pos,      # [batch, n_contact, 3]
    target_contact_normal,   # [batch, n_contact, 3]
    object_pose,             # [batch, 4, 4]
    gpu_memory_pool,
    n_iter=4,
    project_per_n_step=3,
    ret_mesh_buffer=False,
    projection_orientation_weight=0.01,
    error_tolerance=0.004,
    ik_step_size=0.4,
    ik_regularization=2e-4
):
    batch_size = contact_ids.shape[0]

    if batch_size == 0:
        print("No solution found.")
        return 

    v_tensor = torch.from_numpy(mesh['v']).cuda().float()
    f_tensor = torch.from_numpy(mesh['f']).cuda().int()
    n_tensor = torch.from_numpy(mesh['n']).cuda().float()
    v_idx_tensor = torch.from_numpy(mesh['vi']).cuda().int()
    f_idx_tensor = torch.from_numpy(mesh['fi']).cuda().int()

    contact_queries = target_contact_pos
    contact_queries_normal = F.normalize(target_contact_normal, dim=-1)
    q = q_init 

    for i in tqdm(range(n_iter * project_per_n_step), desc="Kinematics Finetuning"):
        mesh_pose = batch_fk(tree, q)["link"]

        if i % project_per_n_step == 0:
            refined_contact = gem.batch_solve_contact_with_normal(
                mesh_pose,
                contact_link_ids.int(),
                v_tensor,
                v_idx_tensor,
                f_tensor,
                n_tensor,
                f_idx_tensor,
                contact_queries,
                contact_queries_normal,
                True,    # to link frames.
                projection_orientation_weight
            )

            next_contact_pos_in_linkf = refined_contact[:, :, :3]
            next_contact_normal_in_linkf = refined_contact[:, :, 3:]

            contact_pos_in_linkf = next_contact_pos_in_linkf
            contact_normal_in_linkf = next_contact_normal_in_linkf
 
        ik_result = batch_contact_ik(
            tree,
            target_contact_pos, 
            target_contact_normal, 
            contact_pos_in_linkf, 
            contact_normal_in_linkf, 
            contact_link_ids,   
            n_retry=1, 
            step_size=ik_step_size, 
            max_iter=1, 
            regularization=ik_regularization,
            single_update_mode=True,
            q_init=q,
            all_joint_result_buffer=gpu_memory_pool.get_ik_joint_buffer(q_init.size(0), 1),
            all_link_result_buffer=gpu_memory_pool.get_ik_link_buffer(q_init.size(0), 1),
            jac_result_buffer=gpu_memory_pool.get_ik_jac_result_buffer(q_init.size(0), 1),
            J_err_buffer=gpu_memory_pool.get_ik_jac_error_buffer(q_init.size(0), 1, contact_link_ids.size(-1))
        )
        q = ik_result["q"]

    mesh_pose = batch_fk(tree, q)["link"]
    refined_contact = gem.batch_solve_contact_with_normal(
        mesh_pose,
        contact_link_ids.int(),
        v_tensor,
        v_idx_tensor,
        f_tensor,
        n_tensor,
        f_idx_tensor,
        contact_queries,
        contact_queries_normal,
        True,    # to link frames.
        0.0
    )

    contact_pos_in_linkf = refined_contact[:, :, :3]
    contact_normal_in_linkf = refined_contact[:, :, 3:]

    # Filter.
    err = ik_result["err"]
    success_idx = (err < error_tolerance ** 2).all(dim=-1)
 
    # q:            [batch, n_dof]
    # success_idx:  [batch]
    result = {
        "q":                q[success_idx], 
        "q_mask":           q_mask[success_idx],
        "object_pose":      object_pose[success_idx],
        "target_pos":       target_contact_pos[success_idx],
        "target_normal":    target_contact_normal[success_idx],
        "contact_pos":      contact_pos_in_linkf[success_idx],
        "contact_normal":   contact_normal_in_linkf[success_idx],
        "contact_link_id":  contact_link_ids[success_idx],
        
    }

    if ret_mesh_buffer:
        result["mesh_buffer"] = {
            "v": v_tensor,
            "f": f_tensor,
            "n": n_tensor,
            "vi": v_idx_tensor,
            "fi": f_idx_tensor
        }
    return result