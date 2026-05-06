# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra import lbvh 
from pathlib import Path
import importlib.util
import torch 
import torch.nn.functional as F 
import time 


def spread_bits(n):
    n = n.int()
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n
    

def morton_3d(x, y, z, bits=10):
    """
    Interleave bits of 3 integers to compute 3D Morton code.
    Assumes x, y, z are integer tensors in [0, 2^bits - 1]
    """

    mask = (1 << bits) - 1
    x = x & mask
    y = y & mask
    z = z & mask
    
    xx = spread_bits(x)
    yy = spread_bits(y)
    zz = spread_bits(z)
    
    return (xx << 2) | (yy << 1) | zz

def morton_sort(tensor, bits=10):
    coords = tensor[:, :3]

    # Normalize
    min_coords = coords.min(dim=0).values
    max_coords = coords.max(dim=0).values
    normalized = (coords - min_coords) / (max_coords - min_coords + 1e-9)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # Morton Code
    quantized = (normalized * ((1 << bits) - 1)).to(torch.int32)
    x, y, z = quantized[:, 0], quantized[:, 1], quantized[:, 2]
    morton_codes = morton_3d(x, y, z, bits=bits)

    # Sort
    sorted_indices = torch.argsort(morton_codes)
    return tensor[sorted_indices], sorted_indices
    

class ContactFieldS2BundleLBVH:
    def __init__(self):
        self.bvh = None

    def build(self, box_min, box_max, all_point, all_point_dir, all_id, leaf_size=128):
        self.bvh = lbvh.lux_build_s2bundle_bvh(
            box_min.float(), 
            box_max.float(), 
            all_point.float(),
            all_point_dir.float(),
            all_id.int(),
            leaf_size
        )
        return 

    def query(self, point, point_dir, angle_threshold=0.6, bvh_buffer=None):
        """
        Args:
        point: [b, 3]
        point_dir: [b, 3]
        """
        n_point = point.shape[0]

        sorted_pos_normal, sorted_indices = morton_sort(torch.cat((point.view(-1, 3), point_dir.view(-1, 3)), dim=-1))
        inverse_indices = torch.argsort(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(len(sorted_indices)).to(sorted_indices.device)
        sorted_object_pos, sorted_object_normal = sorted_pos_normal[:, :3], sorted_pos_normal[:, 3:]

        if bvh_buffer is None:
            print("[WARNING] allocating bvh buffer dynamically. This can be inefficient.")
            bvh_buffer = torch.zeros(n_point * 8, dtype=torch.int32).cuda()

        mask = lbvh.lux_traverse_s2bundle_bvh_with_buffer(
            self.bvh, 
            sorted_object_pos.clone().float(),    # must do clone() to get the right result. I have no idea why we need this.
            sorted_object_normal.clone().float(),
            bvh_buffer,
            angle_threshold
        ).int()
        torch.cuda.synchronize()
        return mask[inverse_indices]
