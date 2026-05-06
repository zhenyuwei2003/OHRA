# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np 

class Leap(RobotInterface):
    def get_canonical_space(self):
        box_min = np.array([ 0.08, -0.03, 0.08], dtype=np.float32)
        box_max = np.array([0.12, 0.03, 0.13], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        return './assets/hand/leap/leap_hand_right.urdf'

    def get_contact_field_config(self):
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {}
        }

        for link in [ 
            "base_link", 
            "fingertip_3",  
            "fingertip_2",  
            "fingertip",  
            "thumb_fingertip"
        ]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 1.57)
                ]
            }

        config["static_link"]["base_link"] = {
            "allowed_normal": [
                (np.array([[1.0, 0.0, 0.0]]), 3.1415926 * 0.25)
            ]
        }

        return config

    def get_active_joints(self):
        return [f'{i}' for i in range(16)]

    def get_base_link(self):
        return "base"

    def get_static_links(self):
        return ["palm_lower"]

    def get_mesh_scale(self):
        return 1.0
