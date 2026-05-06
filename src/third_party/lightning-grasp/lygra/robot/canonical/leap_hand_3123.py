import os
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[5])
REPO_DIR = str(Path(__file__).resolve().parents[3])
sys.path.append(REPO_DIR)

from lygra.robot.base import RobotInterface


class Leap3123(RobotInterface):
    def get_canonical_space(self):
        box_min = np.array([0.08, -0.03, 0.08], dtype=np.float32)
        box_max = np.array([0.12, 0.03, 0.13], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        return f'{ROOT_DIR}/grasp_zeroshot/assets/urdf/leap_hand_3123.urdf'

    def get_contact_field_config(self):
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {}
        }

        for link in [
            "thumb_distal",
            "index_proximal",
            "middle_middle",
            "little_distal",
        ]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 1.57)
                ]
            }

        config["static_link"]["base_link"] = {
            "allowed_normal": [
                (np.array([[1.0, 0.0, 0.0]]), 3.1416 * 0.25)
            ]
        }

        return config

    def get_active_joints(self):
        return [
            'thumb_joint1', 'thumb_joint2', 'thumb_joint4', 'thumb_joint5',
            'index_joint1', 'index_joint2',
            'middle_joint1', 'middle_joint2', 'middle_joint3',
            'little_joint1', 'little_joint2', 'little_joint3', 'little_joint4'
        ]

    def get_base_link(self):
        return "world"

    def get_static_links(self):
        return ["palm"]

    def get_mesh_scale(self):
        return 1.0
