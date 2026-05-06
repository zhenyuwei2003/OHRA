# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
from urdfpy import URDF
from pathlib import Path 
import itertools
import open3d as o3d
import torch
from lygra.utils.vis_utils import show_visualization
from lygra.mesh import RobotMesh


class RobotVisualizer:
    def __init__(self, robot):
        self.robot = robot
        self.robot_tree = robot.get_kinematics_tree() 
        self.robot_mesh = RobotMesh(robot.urdf_path,  mesh_scale=robot.get_mesh_scale())
        return 

    def get_mesh_fk(self, q, mode='o3d', visual=True):
        """ Get robot mesh (trimesh format) under configuration q.
        
        Args:
            q: [n_dof], np.ndarray
        
        Returns:
            mesh. trimesh.Trimesh
        """
        from lygra.kinematics import batch_fk

        if not isinstance(q, torch.Tensor):
            q = torch.from_numpy(q).cuda()
        q = q.view(1, -1)
        link_poses = batch_fk(self.robot_tree, q)["link"].detach().cpu().numpy()[0]
        link_poses = {self.robot_tree.links[i]: p for i, p in enumerate(link_poses)}
        meshes = self.robot_mesh.get_all_link_meshes(link_poses, visual=visual)
        
        if mode == 'o3d':
            o3d_meshes = []
            for i, m in enumerate(meshes):
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLitTransparency"
                material.base_color = [1.0, 1.0, 1.0, 0.8]
                material.base_metallic = 0.1
                material.base_roughness = 1.0
                o3d_meshes.append({"name": str(i), "geometry": m, "material": material})
            return o3d_meshes
        
        else:
            mesh = trimesh.util.concatenate(self.robot_mesh.get_all_link_meshes(link_poses, visual=visual, to_o3d=False))
            return mesh

    def show(self, visuals):
        show_visualization(visuals)
