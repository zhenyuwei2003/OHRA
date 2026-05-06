# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np 


class Shadow(RobotInterface):
    def get_canonical_space(self):
        box_min = np.array([-0.01, -0.10, 0.07], dtype=np.float32)
        box_max = np.array([0.03, -0.05, 0.13], dtype=np.float32)
        return box_min, box_max 
    
    def get_default_urdf_path(self):
        return './assets/hand/shadow/shadow_hand_right.urdf'
    
    def get_contact_field_config(self):
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {}
        }

        for link in ["rfdistal", "lfdistal", "thdistal", "mfdistal", "ffdistal"]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 3.1415926/2)
                ]
            }
        
        for link in ["lfmiddle", "rfmiddle", "mfmiddle", "ffmiddle"]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 3.1415926/2),
                    (np.array([0.0, 0.0, 1.0]), 3.1415926/2),
                    (np.array([0.0, 1.0, 0.0]), 3.1415926/4)
                ]
            }
        
        config["static_link"]["base_link"] = {
            "allowed_normal": [
                (np.array([[0.0, -1.0, 0.0]]), 3.1415926 * 0.25)
            ]
        }

        return config

    def get_active_joints(self):
        result = [ 
            "RFJ1", "RFJ2", "RFJ3", "RFJ4", 
            "LFJ1", "LFJ2", "LFJ3", "LFJ4", "LFJ5",
            "THJ1", "THJ2", "THJ3", "THJ4", "THJ5",
            "MFJ1", "MFJ2", "MFJ3", "MFJ4", 
            "FFJ1", "FFJ2", "FFJ3", "FFJ4"
        ]
        return result

    def get_mesh_scale(self):
        return 0.001

    def get_base_link(self):
        return "palm"

    def get_static_links(self):
        return ["palm"]
