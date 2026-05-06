# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.batch_grasp_optimizer import BatchedZerothOrderKinematicGraspOptimizer
import torch


def search_contact_point(
    contact_domain_pos, 
    contact_domain_normal,
    contact_domain_point_idx, 
    object_poses, 
    contact_ids, 
    batch_size=128, 
    return_hand_frame=True,
    condition={},
    zo_step=5,
    zo_lr=0.005,
    threshold=0.15
):
    """
    TODO: Wrap the score function & criterion into a class.
    
    Args:
        TODO: No procrastination please
        ...

    Returns:
        success_contact_pos:        # [n_success, n_contact, 3]
        success_contact_normal:     # [n_success, n_contact, 3]
        success_object_pose:        # [n_success, 4, 4]
        ...

    """
    optimizer = BatchedZerothOrderKinematicGraspOptimizer(contact_domain_pos, contact_domain_normal, contact_domain_point_idx)
    solution = optimizer.init_solution(batch_size)

    if "extra_contact_pos" in condition:
        condition["extra_contact_pos"] = condition["extra_contact_pos"].unsqueeze(1).expand(-1, batch_size, -1, -1)
        condition["extra_contact_normal"] = condition["extra_contact_normal"].unsqueeze(1).expand(-1, batch_size, -1, -1)
        condition["extra_contact_mask"] = condition["extra_contact_mask"].unsqueeze(1).expand(-1, batch_size, -1)

    solution = optimizer.optimize(solution, condition=condition, step=zo_step, zo_lr=zo_lr)

    object_pose_idx, grasp_idx = torch.where(solution["score"] < threshold)

    success_contact_pos = solution["contact_pos"][object_pose_idx, grasp_idx]               # [n_success, n_contact, 3]
    success_contact_normal = solution["contact_normal"][object_pose_idx, grasp_idx]         # [n_success, n_contact, 3]
    success_contact_point_idx = solution["contact_point_idx"][object_pose_idx, grasp_idx]   # [n_success, n_contact]

    success_object_pose = object_poses[object_pose_idx]                                     # [n_success, 4, 4]
    success_contact_ids = contact_ids[object_pose_idx]                                      # [n_success, n_contact]

    if return_hand_frame:
        success_contact_pos = torch.bmm(
            success_contact_pos, 
            success_object_pose[:, :3, :3].permute(0, 2, 1)
        ) + success_object_pose[:, :3, 3].unsqueeze(1)
        
        success_contact_normal = torch.bmm(
            success_contact_normal, 
            success_object_pose[:, :3, :3].permute(0, 2, 1)
        )

    return  success_contact_pos, success_contact_normal, success_contact_point_idx, \
            success_object_pose, success_contact_ids, object_pose_idx
