# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np 


class DClaw(RobotInterface):
    def get_canonical_space(self):
        box_min = np.array([0.16, -0.02, -0.02], dtype=np.float32)
        box_max = np.array([0.22, 0.02, 0.02], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        return './assets/hand/dclaw_gripper/dclaw_gripper.urdf'

    def get_contact_field_config(self):
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {},

            "patch_pos_dist_tol": 0.01,
            "patch_rot_dist_tol": 3.1415926 * 0.2
        }

        for link in ["link_f1_3_tip", "link_f2_3_tip", "link_f3_3_tip"]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 3.1415926/2)
                ]
            }
        
        config["static_link"]["base_link"] =  {
            "allowed_normal": [
                (np.array([[-1.0, 0.0, 0.0]]), 0.0)
            ]
        }

        return config

    def get_active_joints(self):
        result = [ 
            "joint_f1_0", "joint_f1_1", "joint_f1_2", 
            "joint_f2_0", "joint_f2_1", "joint_f2_2", 
            "joint_f3_0", "joint_f3_1", "joint_f3_2" 
        ]
        return result

    def get_mesh_scale(self):
        return 1.0

    def get_base_link(self):
        return "base"

    def get_static_links(self):
        return ["base_link"]
