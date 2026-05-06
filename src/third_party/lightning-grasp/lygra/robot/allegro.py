# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np 


class Allegro(RobotInterface):
    def get_canonical_space(self):
        """
            Canonical Space for placing objects (we randomly select an object surface point and drag it into this box)
        """
        box_min = np.array([0.08, -0.03, -0.04], dtype=np.float32)
        box_max = np.array([0.13, 0.03, 0.04], dtype=np.float32)
        return box_min, box_max 

    def get_default_urdf_path(self):
        """
            Default path to your robot URDF
        """
        return './assets/hand/allegro_right/allegro_hand_right.urdf'

    def get_contact_field_config(self):
        """
            Specify which links should make contact:
            - Static Links: e.g. palms.
            - Movable Links: e.g. fingers.

            For movable links, you can restrict contact normal directions using the format below.

        """

        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {}
        }

        for link in [
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_15.0_tip"
        ]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([0.0, 0.0, -1.0]), 3.1415926 * 0.49999)
                ]
            }

        config["static_link"]["base_link"] = {
            "allowed_normal": [
                (np.array([[1.0, 0.0, 0.0]]), 3.1415926 * 0.25)
            ]
        }
        return config


    def get_active_joints(self):
        """
            Specify the active joints (dofs).
            Our system will return active joint values in this order.
        """
        return [f'joint_{i}.0' for i in range(16)]

    def get_base_link(self):
        return "base_link"

    def get_static_links(self):
        return ["base_link"]

    def get_mesh_scale(self):
        """
            Your robot mesh might be rescaled in your URDF, specify it here.
            (will be removed in the future.)
        """
        return 1.0


