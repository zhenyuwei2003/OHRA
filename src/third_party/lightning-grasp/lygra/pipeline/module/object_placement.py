# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np 
from lygra.utils.geom_utils import get_tangent_plane
from lygra.pipeline.module.collision import batch_object_hand_collision_check


def get_object_pose_sampling_args(strategy, robot):
    args = {
        "strategy": strategy
    }

    if strategy == 'canonical':
        bmin, bmax = robot.get_canonical_space()
        args['bmin'] = bmin 
        args['bmax'] = bmax

    return args


def get_align_transform(obj_pos, obj_normal, robot_pos, robot_normal):
    obj_transform = np.eye(4)
    obj_x, obj_y = get_tangent_plane(obj_normal)
    obj_transform[:3, 0] = obj_x 
    obj_transform[:3, 1] = obj_y 
    obj_transform[:3, 2] = obj_normal
    obj_transform[:3, 3] = obj_pos 

    robot_transform = np.eye(4)
    robot_x, robot_y = get_tangent_plane(-robot_normal)
    robot_transform[:3, 0] = robot_x
    robot_transform[:3, 1] = robot_y 
    robot_transform[:3, 2] = -robot_normal 
    robot_transform[:3, 3] = robot_pos
    return robot_transform @ np.linalg.inv(obj_transform)


def sample_object_pose_kernel(
    n, 
    points, 
    normals, 
    contact_field,
    tree, 
    mesh_data, 
    sampling_args,
    use_static_prob=0.5
):
    """
    Args:
        points:  [N_point, 3]
        normals: [N_point, 3]
        ...

    Returns:
        ...
    """

    result = []
    result_extra_contact_pos = []
    result_extra_contact_normal = []
    result_extra_contact_mask = []

    n_static = int(n * use_static_prob)
    n_non_static = n - n_static

    robot_contact = contact_field.sample_spatial_contact(n_non_static, sampling_args)
    
    sample_idx = np.random.randint(0, len(points), n_non_static)
    sampled_object_contact_points = points[sample_idx]
    sampled_object_contact_normals = normals[sample_idx]

    """
    TODO: rewrite all these nonsense
    """


    if isinstance(sampled_object_contact_normals, torch.Tensor):
        sampled_object_contact_points = sampled_object_contact_points.detach().cpu().numpy()
        sampled_object_contact_normals = sampled_object_contact_normals.detach().cpu().numpy()

    for i in range(n_non_static):
        transform = get_align_transform(
            sampled_object_contact_points[i],
            sampled_object_contact_normals[i],
            robot_contact[i, :3],
            robot_contact[i, 3:]
        )
        result.append(transform)
        result_extra_contact_mask.append(np.array([0.0]))
        result_extra_contact_pos.append(np.zeros((1, 3)))
        result_extra_contact_normal.append(np.zeros((1, 3)))

    # we also need to filter these.
    if n_static > 0 and contact_field.sample_static_contact_geometry(1) is not None:
        sample_idx = np.random.randint(0, len(points), n_static)
        sampled_object_contact_points = points[sample_idx]
        sampled_object_contact_normals = normals[sample_idx]

        if isinstance(sampled_object_contact_normals, torch.Tensor):
            sampled_object_contact_points = sampled_object_contact_points.detach().cpu().numpy()
            sampled_object_contact_normals = sampled_object_contact_normals.detach().cpu().numpy()

        robot_contact = contact_field.sample_static_contact_geometry(n_static)

        for i in range(n_static):
            result.append(
                get_align_transform(
                    sampled_object_contact_points[i],
                    sampled_object_contact_normals[i],
                    robot_contact[i, :3],
                    robot_contact[i, 3:]
                )
            )
            result_extra_contact_mask.append(np.array([1.0]))
            result_extra_contact_pos.append(robot_contact[i:i+1, :3])
            result_extra_contact_normal.append(robot_contact[i:i+1, 3:])

    # Ensure no penetration
    result = torch.from_numpy(np.array(result)).cuda().float() # [n, 4, 4]
    
    points = points.unsqueeze(0).expand(result.shape[0], -1, -1)
    rotation = result[:, :3, :3]
    translation = result[:, :3, 3]  # [n, 3]

    point_transformed = torch.bmm(points, rotation.transpose(-1, -2)) + translation.unsqueeze(1)    # [n, n_point, 3]

    no_collision = batch_object_hand_collision_check(
        tree=tree,
        mesh=mesh_data,
        object_point=point_transformed
    )

    selected_idx = torch.where(no_collision)
    out = result[selected_idx]

    condition = {
        "extra_contact_pos": torch.from_numpy(np.array(result_extra_contact_pos)).cuda()[selected_idx].float(),
        "extra_contact_normal": torch.from_numpy(np.array(result_extra_contact_normal)).cuda()[selected_idx].float(),
        "extra_contact_mask": torch.from_numpy(np.array(result_extra_contact_mask)).cuda()[selected_idx].float()
    }
    return out, condition


def sample_object_pose(
    n, 
    points, 
    normals, 
    contact_field, 
    tree, 
    mesh_data, 
    sampling_args, 
    use_static_prob=0.5
):
    """
    TODO(): batch it please.
    Args:
        points:  [N_point, 3]
        normals: [N_point, 3]
    
    Returns:
        ...
    """

    n_sol = 0
    all_result_pose = []
    all_result_condition = None

    while n_sol < n:
        pose, condition = sample_object_pose_kernel(
            n, points, normals, 
            contact_field, tree, mesh_data, 
            sampling_args=sampling_args,
            use_static_prob=use_static_prob
        )

        all_result_pose.append(pose)
        if all_result_condition is None:
            all_result_condition = condition
        else:
            for k, v in all_result_condition.items():
                all_result_condition[k] = torch.cat((v, condition[k]), dim=0)

        n_sol += len(pose)

    all_result_pose = torch.cat(all_result_pose, dim=0)
    return all_result_pose, all_result_condition
