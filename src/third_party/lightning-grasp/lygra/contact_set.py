# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 


def sample_from_mask(mask):
    """
    Args:
        mask. [B, M]. binary mask.

    Returns:
        sampled [B,]. index matrix. Uniformly sampled index in each row whose value is 1. Set to -1 for all-zero row. 
    """
    mask = mask.bool()
    B, M = mask.shape
    all_indices = torch.arange(M).expand(B, M).to(mask.device)
    masked_indices = all_indices.masked_fill(~mask, M)
    valid_counts = mask.sum(dim=1)
    rand_pos = torch.randint(0, valid_counts.max().clamp(min=1), (B,)).to(mask.device)
    rand_pos = torch.minimum(rand_pos, valid_counts - 1)
    sorted_masked, _ = masked_indices.sort(dim=1)
    sampled = sorted_masked[torch.arange(B).to(mask.device), rand_pos]
    sampled[valid_counts == 0] = -1
    return sampled 


def sample_independent_set(n, initial_mask, dependency_matrix):
    """
    Sample a independent n-subset from a set of M objects.

    Args:
        initial_mask:       [B, M]
        dependency_matrix:    [M, M]

    """

    B = initial_mask.shape[0]
    M = dependency_matrix.shape[0]

    mask = initial_mask.clone()

    all_samples = []

    for i in range(n):
        sampled = sample_from_mask(mask)  # [B]
        # update the mask using dependency matrix.

        # indexing, returning Y [B, M] 
        # Y[i][j] = D[sampled[i]][j]

        # Advanced Indexing
        # Y[i][j] = D[I1[i][j], I2[i][j]]
        # ==>   I1[i][j] == sampled[i]
        # ==>   I2[i][j] == j 

        sampled_ = torch.clip(sampled, 0, M).long()
        I1 = sampled_.unsqueeze(-1).expand(-1, M)
        I2 = torch.arange(M).unsqueeze(0).expand(B, -1).to(initial_mask.device)

        update_mask = 1 - dependency_matrix[I1, I2]

        mask = mask * update_mask
        all_samples.append(sampled)
    
    all_samples = torch.stack(all_samples, dim=-1)
    return all_samples 


def get_dependency_matrix(contact_field, dependency_sets):
    """

    Returns:
        D[i,j] = 1. if contact patch i and j are not independent. (e.g. on the same finger) else 0
    """
    names = contact_field.get_all_parent_link_names() 

    dependency_matrix = torch.zeros(len(names), len(names))
    for ds in dependency_sets:
        all_idx = torch.tensor([i for i, x in enumerate(names) if x in ds])
        dependency_matrix[all_idx.unsqueeze(1), all_idx] = 1
    
    return dependency_matrix


def get_link_dependency_matrix(contact_field, dependency_sets):
    """

    Returns:
        D[i,j] = 1. if contact candidate link i and j are not independent. (e.g. on the same finger) else 0
    """
    names = contact_field.get_all_contact_link_names()

    dependency_matrix = torch.zeros(len(names), len(names))
    for ds in dependency_sets:
        all_idx = torch.tensor([i for i, x in enumerate(names) if x in ds])
        dependency_matrix[all_idx.unsqueeze(1), all_idx] = 1

    return dependency_matrix

