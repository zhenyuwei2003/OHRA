# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
from lygra import gem
from lygra.contact_set import get_dependency_matrix, get_link_dependency_matrix, sample_independent_set


def sample_from_mask_and_gather(mask, data, num_sample):
    """
    Args:
        mask:   [B, K, N] binary mask (0/1)
        data:   [N, 3/6] data tensor
        num_sample: int. number of indices to sample per [B, K]
    
    Returns:
        gathered:       [B, K, num_sample, 3/6] gathered from `data`
        gathered_idx:   [B, K, num_sample]
    """
    out, out_idx = gem.batch_random_mask_gather(mask.int(), data.float(), num_sample)
    return out, out_idx


def sample_pose_and_contact_from_interaction(
    n_contact, 
    interaction_matrix, 
    dependency_matrix, 
    object_points, 
    object_normals, 
    object_poses, 
    condition={},
    contact_domain_resolution=500,
    n_sample_repeat=4,
    min_contact_domain_size=5
):
    """ This function returns randomly sampled, valid contact domains with their associated object poses.

    Args:
        n_contact: 
        interaction_matrix:         [n_pose, n_contact_field, n_object_point]. This is a boolean tensor. 
        -  Note: interaction_matrix[i][j][k] indicates whether (object_points[k], object_normals[k]) interact with contact_field[j] when the object pose is object_poses[i]
        
        dependency_matrix:          [n_contact_field, n_contact_field]
        -  Note: 

        object_points:              [n_object_point, 3]
        object_normals:             [n_object_point, 3]
        object_poses:               [n_object_point, 4, 4]
        contact_domain_resolution:  int. The resolution of returned contact domain.
    
    Returns:
        contact_domain_pos:         [n_valid_pose, n_contact, contact_domain_resolution, 3]
        contact_domain_normal:      [n_valid_pose, n_contact, contact_domain_resolution, 3]
        object_poses:               [n_valid_pose, 4, 4]
        contact_ids:                [n_valid_pose, n_contact]
    """
    assert min_contact_domain_size > 0 

    interaction_score = interaction_matrix.sum(dim=-1)                                              # [n_batch_outer, n_contact_field]
    initial_mask = torch.zeros_like(interaction_score)
    initial_mask[torch.where(interaction_score > min_contact_domain_size)] = 1

    success = False
    while n_contact >= 2:
        contact_ids = sample_independent_set(n_contact, initial_mask, dependency_matrix)                # [n_batch_outer, n_contact]
    
        valid_batch_idx = torch.where((contact_ids >= 0).all(dim=-1))
        if len(valid_batch_idx[0]) > 0:
            break
        else:
            # too hard, resample.
            print("Resampling")
            n_contact -= 1 
            if n_contact == 1:
                print("Search Failed")
                sys.exit(-1)

    contact_ids = contact_ids[valid_batch_idx]                                                      # [n_valid_pose, n_contact]
    interaction_matrix = interaction_matrix[valid_batch_idx]

    object_poses = object_poses[valid_batch_idx]  # [B, 4, 4]
    filtered_condition = {k: v[valid_batch_idx] for k, v in condition.items()}

    batch_indices = torch.arange(contact_ids.size(0)).unsqueeze(1).expand(-1, contact_ids.size(1))  # [B, N_contact]
    contact_domain_mask = interaction_matrix[batch_indices, contact_ids]                            # [B, N_contact, N]
    assert contact_domain_mask.any(dim=-1).all(), "Contact domain must have at least one element."
 
    gathered, contact_domain_idx = sample_from_mask_and_gather(contact_domain_mask, torch.cat((object_points, object_normals), dim=-1), contact_domain_resolution)
    contact_domain_pos = gathered[..., :3]
    contact_domain_normal = gathered[..., 3:]

    return contact_domain_pos, contact_domain_normal, contact_domain_idx, object_poses, contact_ids, filtered_condition, valid_batch_idx
