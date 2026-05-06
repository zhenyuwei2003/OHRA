from lygra.bvh_s2bundle import ContactFieldS2BundleLBVH
from lygra.kinematics  import batch_fk
from lygra.utils.geom_utils import get_tangent_plane
from lygra.utils.vis_utils import trimesh_to_open3d, make_arrow_lines, show_visualization
from tqdm import tqdm
import numpy as np 
import torch
import math 
import torch.nn.functional as F 


def create_cube_boxes_vectorized(points, cube_width):
    points = points.float()
    min_bound = points.min(dim=0).values
    max_bound = points.max(dim=0).values
    min_bound = min_bound - cube_width * 0.1
    max_bound = max_bound + cube_width * 0.1
    
    grid_coords = ((points - min_bound) / cube_width).floor().long()
    unique_cubes = torch.unique(grid_coords, dim=0)
    cubes_min = min_bound + unique_cubes.float() * cube_width
    cubes_max = cubes_min + cube_width
    return cubes_min, cubes_max


def sample_points_from_box_union(all_box_min, all_box_max, b):
    """ Sample `b` points uniformly from the union of axis-aligned 3D boxes.

    Args:
        all_box_min: (N, 3)
        all_box_max: (N, 3)
        b: int, number of points to sample.

    Returns:
        points: (b, 3)
    """

    box_volumes = torch.prod(all_box_max - all_box_min, dim=1)
    probs = box_volumes / box_volumes.sum()
    chosen_boxes = torch.multinomial(probs, num_samples=b, replacement=True)
    rand = torch.rand(b, 3, device=all_box_min.device)
    points = all_box_min[chosen_boxes] + rand * (all_box_max[chosen_boxes] - all_box_min[chosen_boxes])
    return points


class LBVHS2ContactFieldAccelStructure:
    def __init__(self, origin, direction, ids, angle_range, box_w=0.01, max_leaf_size=256):
        self.contact_lbvhs = [] 
        self.all_boxes_min = []
        self.all_boxes_max = []
        self.all_angle_range = angle_range

        for i in tqdm(range(len(origin)), desc="Building Contact Field BVH"):
            o = torch.from_numpy(origin[i]).cuda()
            d = torch.from_numpy(direction[i]).cuda()
            boxes_min, boxes_max = create_cube_boxes_vectorized(o, box_w)

            bvh = ContactFieldS2BundleLBVH()
            bvh.build(boxes_min, boxes_max, o, d, torch.from_numpy(ids[i]).cuda().int(), max_leaf_size)

            self.contact_lbvhs.append(bvh)
            self.all_boxes_min.append(boxes_min)
            self.all_boxes_max.append(boxes_max)

        self.all_boxes_min = torch.cat(self.all_boxes_min)
        self.all_boxes_max = torch.cat(self.all_boxes_max)
        self.center = (self.all_boxes_min.mean(dim=0) + self.all_boxes_max.mean(dim=0)) / 2
        self.device = self.all_boxes_max.device 

    def sample_spatial_contact(self, n):
        point = sample_points_from_box_union(self.all_boxes_min, self.all_boxes_max, n)

        normal = torch.randn(n, 3).to(self.device)
        normal = F.normalize(normal, dim=-1)
        return np.concatenate([point.detach().cpu().numpy(), normal.detach().cpu().numpy()], axis=-1)

    def compute_interaction(self, object_pos, object_normal, angle_threshold=0.2):
        """
        Args: 
        object_pos:     [b, n, 3],   torch.Tensor
        object_normal:  [b, n, 3],   torch.Tensor

        Returns:
        result (box hit id; -1 no hit):       [b, n_field, n_point],  torch.Tensor
        """
        b = object_pos.shape[0]
        n_obj = object_pos.shape[1]

        # Output
        all_result = []

        # Use preallocated bvh buffer.
        bvh_buffer = torch.zeros(object_pos.view(-1, 3).shape[0] * 8, dtype=torch.int32).cuda()
        for i, bvh in enumerate(tqdm(self.contact_lbvhs, desc="Contact Field BVH Traversal")):
            bvh_buffer = bvh_buffer * 0
            result = bvh.query(
                object_pos.view(-1, 3), 
                -object_normal.view(-1, 3), 
                angle_threshold=angle_threshold,
                bvh_buffer=bvh_buffer
            )  # [1, b * nobj]
            all_result.append(result)

        all_result = torch.cat(all_result, dim=0)
        all_result = all_result.view(-1, b, n_obj).permute(1, 0, 2)
        return all_result
