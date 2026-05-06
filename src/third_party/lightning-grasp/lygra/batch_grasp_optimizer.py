# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import scipy
import numpy as np
import time
import torch
import torch.nn.functional as F 
from tqdm import tqdm 
from lygra.utils.geom_utils import get_tangent_plane 


def batched_nnls(A, b, num_iters=15, lr=1.5e-1, init_sol=None):
    """
    Solve a batch of Nonnegative Least Square (NNLS) problem.
    Minimize |Ax - b|^2 subject to x >= 0.

    Args:
    A:      (B, m, n)
    b:      (B, m)
    
    Returns:
    x:      (B, n)
    val:    (B,)
    """
    B, m, n = A.shape
    if init_sol is None:
        x = torch.zeros(B, n, device=A.device)
    else:
        x = init_sol

    At = A.transpose(1, 2)               # (B, n, m)
    AtA = torch.bmm(At, A)               # (B, n, n)
    Atb = torch.bmm(At, b.unsqueeze(2))  # (B, n, 1)

    x = x.unsqueeze(2)
    for i in range(num_iters):
        # TODO(): Maybe fused into a kernel, but the gain can be marginal.
        # In-place implementation of:
        # x = x - lr * (torch.bmm(AtA, x.unsqueeze(2)) - Atb).squeeze(2)
        tmp = torch.bmm(AtA, x)  # shape: (B, N, 1)
        tmp.sub_(Atb)
        tmp.mul_(lr)
        x.sub_(tmp)
        x.clamp_(min=0)

    val = torch.sqrt(((torch.bmm(A, x).squeeze(2) - b) ** 2).sum(dim=-1))
    return x.squeeze(2), val


def compute_score(contact_pos, contact_normal):
    # construct and solve nnls problems.
    n = contact_pos.shape[1]
    best_score = None 

    for i in range(n):
        b = - contact_normal[:, 0, :]
        A = contact_normal[:, 1:, :].permute(0, 2, 1)
        _, score = batched_nnls(A, b)

        if best_score is None:
            best_score = score 
        else:
            best_score = torch.where(score < best_score, score, best_score)

        contact_normal = torch.cat((contact_normal[:, 1:], contact_normal[:, 0:1]), dim=1)
        contact_pos = torch.cat((contact_pos[:, 1:], contact_pos[:, 0:1]), dim=1)
    return best_score


def compute_wrench_score(contact_pos, contact_normal):
    '''
    Args:
        contact_pos:    [..., K, 3]
        contact_normal: [..., K, 3]

    Returns:
        score:          [...]
    '''

    # construct and solve nnls problems.
    leading_shape = contact_pos.shape[:-2]
    K = contact_pos.shape[-2]
    contact_pos = contact_pos.reshape(-1, K, 3)
    contact_normal = contact_normal.reshape(-1, K, 3)

    n = contact_pos.shape[1]
    best_score = None 

    nnls_x_placeholder = None
    for i in range(n):
        b_nf = - contact_normal[:, 0, :]
        A_nf = contact_normal[:, 1:, :].permute(0, 2, 1)            # [b, 3, n]
        
        x = contact_pos[:, 1:, :] - contact_pos[:, 0:1, :]
        f_unit = contact_normal[:, 1:, :]
        wrench = torch.cross(x, f_unit, dim=-1).permute(0, 2, 1)    # [b, 3, n] 

        b_nw = torch.zeros_like(b_nf)
        A_nw = wrench

        A = torch.cat((A_nf, A_nw), dim=1)
        b = torch.cat((b_nf, b_nw), dim=1)

        if nnls_x_placeholder is None:
            B, m, n = A.shape
            nnls_x_placeholder = torch.zeros(B, n, device=A.device)

        _, score = batched_nnls(A, b, x_placeholder=nnls_x_placeholder)
        if best_score is None:
            best_score = score 
        else:
            best_score = torch.where(score < best_score, score, best_score)

        contact_normal = torch.cat((contact_normal[:, 1:], contact_normal[:, 0:1]), dim=1)
        contact_pos = torch.cat((contact_pos[:, 1:], contact_pos[:, 0:1]), dim=1)
    
    best_score = best_score.reshape(*leading_shape)
    return best_score


def compute_wrench_score_vectorized(
    contact_pos, 
    contact_normal, 
    valid_mask=None, 
    init_sol=None, 
    wrench_alpha=10
):
    '''
    Args:
        contact_pos:    [..., n, 3]
        contact_normal: [..., n, 3]
        valid_mask:     [..., n]        which contact vectors should be used in optimization (val=1).

    Returns:
        score:          [...]
    '''

    # construct and solve nnls prqoblems.
    leading_shape = contact_pos.shape[:-2]
    n = contact_pos.shape[-2]
    contact_pos = contact_pos.reshape(-1, n, 3)
    contact_normal = contact_normal.reshape(-1, n, 3)

    all_A = []
    all_b = []

    for i in range(n):
        b_nf = - contact_normal[:, 0, :]
        A_nf = contact_normal[:, 1:, :].permute(0, 2, 1)            # [b, 3, n]
        
        x = contact_pos[:, 1:, :] - contact_pos[:, 0:1, :]
        f_unit = contact_normal[:, 1:, :]
        wrench = torch.cross(x, f_unit, dim=-1).permute(0, 2, 1)    # [b, 3, n] 

        b_nw = torch.zeros_like(b_nf)
        A_nw = wrench * wrench_alpha

        A = torch.cat((A_nf, A_nw), dim=1)          # [b, 6, n-1]
        b = torch.cat((b_nf, b_nw), dim=1)          # [b, 6, n-1]

        all_A.append(A.unsqueeze(1))                # [b, 1, 6, n - 1]
        all_b.append(b.unsqueeze(1))

        contact_normal = torch.cat((contact_normal[:, 1:], contact_normal[:, 0:1]), dim=1)
        contact_pos = torch.cat((contact_pos[:, 1:], contact_pos[:, 0:1]), dim=1)

    all_A = torch.cat(all_A, dim=1)
    all_A = all_A.view(-1, *all_A.shape[2:])
    all_b = torch.cat(all_b, dim=1)
    all_b = all_b.view(-1, *all_b.shape[2:])

    # t = time.time()
    if init_sol is not None:
        nnls_num_iters = 4
    else:
        nnls_num_iters = 15

    sol, score = batched_nnls(all_A, all_b, num_iters=nnls_num_iters, init_sol=init_sol)   # [b * n]
    score = score.view(-1, n)
    torch.cuda.synchronize()

    if valid_mask is not None:
        valid_mask = valid_mask.view(-1, n)  
        # fill the invalid score with a large value.  
        score = score * valid_mask + 1e10 * torch.ones_like(score) * (1 - valid_mask)
    
    best_score = torch.min(score, dim=-1).values
    return best_score.view(*leading_shape), sol


# Compute pairwise distances using broadcasting:
def concentration_score(points):
    diff = points.unsqueeze(-2) - points.unsqueeze(-3)  # shape [..., K, K, 3]
    dists = torch.norm(diff, dim=-1)  # [..., K, K]
    mask = torch.eye(points.shape[-2], device=points.device).bool()
    dists = dists.masked_fill(mask, float('inf'))
    min_dist, _ = dists.min(dim=-1)
    score, _ = min_dist.min(dim=-1)
    score = torch.where(score < 0.008, 0.008 / (score + 0.001), torch.zeros_like(score))
    return score

def project(
    contact_pos,      # [B_out, B_in, N, 3]
    object_pos,       # [B_out, N, M, 3]
    object_normal,    # [B_out, N, M, 3]
    object_point_idx  
):
    ''' Batched Point Projection onto Point Cloud
    
    Args:
        contact_pos:
        object_pos:
        object_normal:
        object_point_idx:

    Return:
        closest_point
        -closest_normal         note that this is negated outer normal.
        closest_point_idx

    '''

    B, B_in, N, _ = contact_pos.shape
    M = object_pos.shape[2]

    contact_flat = contact_pos.permute(0, 2, 1, 3).reshape(B * N, B_in, 3)

    object_flat = object_pos.view(B * N, M, 3)
    normal_flat = object_normal.view(B * N, M, 3)
    if object_point_idx is not None:
        object_point_idx_flat = object_point_idx.view(B*N, M)

    dists = torch.cdist(contact_flat, object_flat, p=2)
    min_indices = dists.argmin(dim=-1)

    batch_indices = torch.arange(B * N, device=contact_pos.device).unsqueeze(1).expand(-1, B_in)
    closest_point_flat = object_flat[batch_indices, min_indices]
    closest_normal_flat = normal_flat[batch_indices, min_indices]
    closest_point = closest_point_flat.view(B, N, B_in, 3).permute(0, 2, 1, 3).contiguous()
    closest_normal = closest_normal_flat.view(B, N, B_in, 3).permute(0, 2, 1, 3).contiguous()
   
    if object_point_idx is not None:
        closest_point_idx_flat = object_point_idx_flat[batch_indices, min_indices]
        closest_point_idx = closest_point_idx_flat.view(B, N, B_in).permute(0, 2, 1).contiguous()
    else:
        closest_point_idx = None

    return closest_point, -closest_normal, closest_point_idx


def optimize_step(
    contact_pos, 
    contact_normal, 
    contact_point_idx,
    object_pos, 
    object_normal, 
    object_point_idx,
    pivot_idx, 
    n_variant=30, 
    step_size=0.01,
    init_contact_score=None,
    init_nnls_sol=None,
    extra_contact_pos=None,
    extra_contact_normal=None,
    extra_contact_mask=None
):
    '''
    Args:
        Contact Points:  [B, B_in, K, 3]
        Contact Normals: [B, B_in, K, 3]
        Object  Points:  [B, K, M, 3]
        Object  Normals: [B, K, M, 3]

        extra_contact_pos:      [B, B_in, K', 3]
        extra_contact_normal:   [B, B_in, K', 3]
        extra_contact_mask:     [B, B_in, K']

        pivot_idx: which i in [N] to optimize.
    
    Returns:

    '''

    all_scores = []

    B, B_in = contact_pos.shape[0], contact_pos.shape[1]
    x, y = get_tangent_plane(contact_normal)            # [B, B_in, K, 3], [B, B_in, K, 3]
    
    assert x.shape == y.shape == contact_normal.shape 

    if extra_contact_pos is not None:
        extra_contact_mask = torch.cat(
            (torch.ones(*contact_pos.shape[:-1]).to(extra_contact_mask.device), extra_contact_mask), dim=-1
        )

    best_score = init_contact_score
    if best_score is None:
        if extra_contact_pos is not None:
            best_score, nnls_sol = compute_wrench_score_vectorized(
                torch.cat((contact_pos, extra_contact_pos), dim=-2), 
                torch.cat((contact_normal, extra_contact_normal), dim=-2),
                extra_contact_mask.contiguous(),
                init_sol=init_nnls_sol
            )

        else:
            best_score, nnls_sol = compute_wrench_score_vectorized(contact_pos, contact_normal)

    else:
        nnls_sol = None

    assert best_score.shape == contact_pos.shape[:2]

    best_contact_pos = contact_pos.clone()
    best_contact_normal = contact_normal.clone()

    if contact_point_idx is not None:
        best_contact_point_idx = contact_point_idx.clone()

    for i in range(n_variant):
        next_contact_pos = contact_pos.clone()
        xy_direction = torch.randn(B, B_in, 2).to(contact_pos.device)

        # Ideally, the stepsize should depend on curvature.
        next_contact_pos[:, :, pivot_idx] += xy_direction[:, :, 0:1] * x[:, :, pivot_idx] * step_size 
        next_contact_pos[:, :, pivot_idx] += xy_direction[:, :, 1:2] * y[:, :, pivot_idx] * step_size

        next_contact_pos, next_contact_normal, next_contact_point_idx = project(
            next_contact_pos, object_pos, object_normal, object_point_idx
        )

        if extra_contact_pos is not None:
            score, _ = compute_wrench_score_vectorized(
                torch.cat((next_contact_pos, extra_contact_pos), dim=-2), 
                torch.cat((next_contact_normal, extra_contact_normal), dim=-2),
                extra_contact_mask,
                init_sol=nnls_sol
            )

        else:
            score, _ = compute_wrench_score_vectorized(next_contact_pos, next_contact_normal)

        improve_idx = torch.where(score < best_score)
        best_score[improve_idx] = score[improve_idx]
        best_contact_pos[improve_idx] = next_contact_pos[improve_idx]
        best_contact_normal[improve_idx] = next_contact_normal[improve_idx]
        if contact_point_idx is not None:
            best_contact_point_idx[improve_idx] = next_contact_point_idx[improve_idx]

    return best_contact_pos, best_contact_normal, best_contact_point_idx, best_score, nnls_sol


def optimize(
    contact_pos, 
    contact_normal, 
    contact_point_idx,
    object_pos, 
    object_normal, 
    object_point_idx,
    tot_global_step=1000, 
    zo_step=10,
    zo_lr=0.005,
    extra_contact_pos=None,
    extra_contact_normal=None,
    extra_contact_mask=None
):
    """
    Args:
        Contact Points:  [B, B_in, K, 3]
        Contact Normals: [B, B_in, K, 3]
        Object  Points:  [B, K, M, 3]
        Object  Normals: [B, K, M, 3]

        extra_contact_pos:      [B, B_in, K', 3] or None
        extra_contact_normal:   [B, B_in, K', 3] or None
        extra_contact_mask:     [B, B_in, K'] or None
    
    Returns:
    
    """
    init_nnls_sol = None
    for global_step in tqdm(range(tot_global_step), desc="Contact Optimization"):
        for pivot_idx in range(contact_pos.shape[2]):
            contact_pos, contact_normal, contact_point_idx, score, nnls_sol = optimize_step(
                contact_pos,
                contact_normal,
                contact_point_idx,
                object_pos, 
                object_normal,
                object_point_idx,
                pivot_idx, 
                n_variant=zo_step,
                step_size=zo_lr,
                init_nnls_sol=init_nnls_sol,
                extra_contact_pos=extra_contact_pos,
                extra_contact_normal=extra_contact_normal,
                extra_contact_mask=extra_contact_mask
            )
            init_nnls_sol = nnls_sol

    score, _ = compute_wrench_score_vectorized(contact_pos, contact_normal)

    return contact_pos, contact_normal, contact_point_idx, score


def batch_sample_domain_data(B_prime, X, X_idx=None):
    """ This function is used for sampling initialization contact point in the contact domain.
    
    out[i][j][k] = X[i][k][random_index_in_N]

    Args:
        X: [B, K, N, 3] — source tensor
        B_prime: int — second batch dimension in M

    Returns:
        out: [B, B', K, 3] — randomly sampled values

    """
    B, K, N, D = X.shape  # D = 3

    # Sample random indices in [0, N) -> shape: [B, B', K]
    sample_indices = torch.randint(0, N, (B, B_prime, K), device=X.device)

    # Prepare indexing tensors for advanced indexing
    batch_i = torch.arange(B, device=X.device).view(B, 1, 1).expand(-1, B_prime, K)       # [B, B', K]
    key_k   = torch.arange(K, device=X.device).view(1, 1, K).expand(B, B_prime, -1)       # [B, B', K]

    # Now use advanced indexing to sample from X
    if X_idx is None:
        out = X[batch_i, key_k, sample_indices]  # -> [B, B', K, 3]
        return out
    else:
        out = X[batch_i, key_k, sample_indices]  # -> [B, B', K, 3]
        out_idx = X_idx[batch_i, key_k, sample_indices]
        return out, out_idx


class BatchedZerothOrderKinematicGraspOptimizer:
    def __init__(self, domain_points, domain_normals, contact_domain_point_idxes=None):
        self.domain_points = domain_points.float()      # [B, K, N, 3]      #   torch.from_numpy(object_points).cuda().float()
        self.domain_normals = domain_normals.float()    # [B, K, N, 3]      #   torch.from_numpy(object_normals).cuda().float()
        self.domain_point_idxes = contact_domain_point_idxes.int()
        return 

    def init_solution(self, batch_size): 
        domain_data = torch.cat((self.domain_points, self.domain_normals), dim=-1)
        sampled_data, contact_point_idx = batch_sample_domain_data(batch_size, domain_data, self.domain_point_idxes)

        contact_pos = sampled_data[..., :3]
        contact_normal = sampled_data[..., 3:]

        return {
            "contact_pos": contact_pos,
            "contact_normal": contact_normal, 
            "contact_point_idx": contact_point_idx
        }

    def optimize(self, solution, condition={}, step=20, zo_step=6, zo_lr=0.005):
        contact_pos, contact_normal, contact_point_idx, score = optimize(
            solution["contact_pos"],
            solution["contact_normal"],
            solution["contact_point_idx"],
            self.domain_points,
            self.domain_normals,
            self.domain_point_idxes,
            tot_global_step=step,
            extra_contact_pos=condition.get("extra_contact_pos", None),
            extra_contact_normal=condition.get("extra_contact_normal", None),
            extra_contact_mask=condition.get("extra_contact_mask", None),
            zo_step=zo_step,
            zo_lr=zo_lr
        )

        return {
            "contact_pos": contact_pos,
            "contact_normal": contact_normal,
            "contact_point_idx": contact_point_idx,
            "score": score 
        }
