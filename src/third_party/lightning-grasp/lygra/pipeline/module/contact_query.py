# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.utils.transform_utils import batch_object_transform


def batch_object_all_contact_fields_interaction(
    object_pos, 
    object_normal, 
    object_pose, 
    accel_structure
):
    """
    Args:
        object_pos:     [N, 3]
        object_normal:  [N, 3]
        object_poses:   [B, 4, 4]
        accel_structure.

    Returns:
        [B, K, N].      Interaction results. # [B, n_contact_field, n_object_point]
    """
    object_transformed = batch_object_transform(object_pose, object_pos, object_normal)
    object_pos = object_transformed["pos"]
    object_normal = object_transformed["normal"]
    result = accel_structure.compute_interaction(object_pos, object_normal)
    return result

