# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 
import torch.nn.functional as F
import trimesh
import numpy as np


class MeshObject:
    def __init__(self, obj_path):
        self.mesh = trimesh.load(obj_path)

    def sample_point_and_normal(self, count=2000):
        points, face_indices = trimesh.sample.sample_surface(self.mesh, count=count)
        normals = self.mesh.face_normals[face_indices]
        return points, normals 

    def get_area(self):
        return self.mesh.area 


def get_tangent_plane(batch_vector):
    ''' Get the tangent planes of vectors.
    
    Args:
        batch_vector: [..., 3] (torch.Tensor or np.ndarray)
    
    Returns:
        x:            [..., 3]
        y:            [..., 3]
    '''
    if isinstance(batch_vector, torch.Tensor):
        shape = batch_vector.shape[:-1]
        batch_vector = F.normalize(batch_vector.reshape(-1, 3), dim=-1)
        x = batch_vector + torch.ones_like(batch_vector) * 2
        y = torch.cross(batch_vector, F.normalize(x, dim=-1), dim=-1)
        x = torch.cross(y, batch_vector, dim=-1)

        return x.reshape(*shape, 3), y.reshape(*shape, 3)
    else:
        # numpy
        shape = batch_vector.shape[:-1]
        
        batch_vector = np.reshape(batch_vector, (-1, 3))
        norm = np.linalg.norm(batch_vector, axis=-1, keepdims=True)
        batch_vector = batch_vector / np.clip(norm, 1e-8, None)

        x = batch_vector + 2.0  # broadcast with scalar
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.clip(x_norm, 1e-8, None)

        y = np.cross(batch_vector, x)
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        y = y / np.clip(y_norm, 1e-8, None)

        x = np.cross(y, batch_vector)
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        x = x / np.clip(x_norm, 1e-8, None)

        
        return x.reshape(*shape, 3), y.reshape(*shape, 3)
