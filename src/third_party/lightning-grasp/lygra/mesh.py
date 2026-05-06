# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import itertools
import open3d as o3d
import torch
import numpy as np
import trimesh
from urdfpy import URDF
from pathlib import Path 


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


class RobotMesh:
    def __init__(self, robot_path, mesh_scale=1.0):
        self.robot = URDF.load(robot_path)
        self.robot_root = Path(robot_path).parent
        self.collision_meshes = {}
        self.visual_meshes = {}
        self.mesh_scale = mesh_scale
        return 

    def create_trimesh_from_data(self, data, convex=False):
        geometry = data.geometry
        
        if geometry.box is not None:
            size = geometry.box.size
            return trimesh.creation.box(extents=size)
        
        elif geometry.sphere is not None:
            radius = geometry.sphere.radius
            return trimesh.creation.icosphere(radius=radius)
        
        elif geometry.cylinder is not None:
            radius = geometry.cylinder.radius
            length = geometry.cylinder.length
            return trimesh.creation.cylinder(radius=radius, height=length)
        
        elif geometry.mesh is not None:
            mesh_path = geometry.mesh.filename
            mesh = trimesh.load(self.robot_root / mesh_path)
            mesh.apply_scale(self.mesh_scale)
            if convex:
                return mesh.convex_hull
            else:
                return mesh

        return None

    def get_link_collision_mesh(self, link_name):
        # This function all the link meshes concatenated as a whole. 
        # For collision detection, we need get_link_collision_meshes_decomposed() insteed of this one.

        # Find the link
        if link_name in self.collision_meshes:
            return self.collision_meshes[link_name].copy()
        
        link = next((l for l in self.robot.links if l.name == link_name), None)

        # Extract collision geometry and convert to Trimesh
        if link is not None:
            trimesh_objects = []
            for collision in link.collisions:
                mesh = self.create_trimesh_from_data(collision, convex=True)
                transform = collision.origin
                if transform is None:
                    # If no transform specified, use identity matrix (no transformation)
                    transform = np.eye(4)
              
                mesh.apply_transform(transform)
                if mesh is not None:
                    trimesh_objects.append(mesh)
                else:
                    print("No valid geometry found for collision")
        else:
            print(f'Link {link_name} not found')

        combined_mesh = trimesh.util.concatenate(trimesh_objects)
        self.collision_meshes[link_name] = combined_mesh

        if isinstance(combined_mesh, list):
            # Some trimesh version can return an empty list when trimesh_objects is [].
            combined_mesh = trimesh.Trimesh(vertices=np.array([]).reshape((0, 3)),
                             faces=np.array([]).reshape((0, 3)))

        return combined_mesh.copy()

    
    def get_link_visual_mesh(self, link_name):   
        if link_name in self.visual_meshes:
            return self.visual_meshes[link_name].copy()

        link = next((l for l in self.robot.links if l.name == link_name), None)
        # Extract collision geometry and convert to Trimesh
        if link is not None:
            trimesh_objects = []
            for visual in link.visuals:
                mesh = self.create_trimesh_from_data(visual)
                transform = visual.origin
                if transform is None:
                    # If no transform specified, use identity matrix (no transformation)
                    transform = np.eye(4)
              
                mesh.apply_transform(transform)
                if mesh is not None:
                    trimesh_objects.append(mesh)
                else:
                    print("No valid geometry found for collision")
        else:
            print(f'Link {link_name} not found')

        combined_mesh = trimesh.util.concatenate(trimesh_objects)

        if isinstance(combined_mesh, list):
            # certain trimesh version can return an empty list when trimesh_objects is [].
            combined_mesh = trimesh.Trimesh(vertices=np.array([]).reshape((0, 3)),
                             faces=np.array([]).reshape((0, 3)))


        self.visual_meshes[link_name] = combined_mesh
        return combined_mesh.copy()

    
    def get_collision_mesh_in_world(self, link_name, pose):
        mesh = self.get_link_collision_mesh(link_name).copy()
        mesh.apply_transform(pose)
        return mesh
    
    def get_link_collision_meshes_decomposed(self, link_name):
        # this is used for collision detection. we need decomposed list.
        # Find the link
        if link_name in self.collision_meshes:
            return self.collision_meshes[link_name].copy()
        
        link = next((l for l in self.robot.links if l.name == link_name), None)

        # Extract collision geometry and convert to Trimesh
        if link is not None:
            trimesh_objects = []
            for collision in link.collisions:
                mesh = self.create_trimesh_from_data(collision, convex=True)
                transform = collision.origin
                if transform is None:
                    # If no transform specified, use identity matrix (no transformation)
                    transform = np.eye(4)
              
                mesh.apply_transform(transform)
                if mesh is not None:
                    trimesh_objects.append(mesh)
                else:
                    print("No valid geometry found for collision")
        else:
            print(f'Link {link_name} not found')

        if len(trimesh_objects) == 0:
            mesh = trimesh.util.concatenate(trimesh_objects)

            if isinstance(mesh, list):
                # certain trimesh version can return an empty list when trimesh_objects is [].
                mesh = trimesh.Trimesh(vertices=np.array([]).reshape((0, 3)),
                                faces=np.array([]).reshape((0, 3)))

            return [mesh]

        else:
            return trimesh_objects


    def get_all_link_meshes(self, link_poses, to_o3d=True, visual=False):
        all_meshes = []
        for link_name, link_pose in link_poses.items():
            # For convenience, link pose is a 4*4 transform matrix. .
            if visual:
                mesh = self.get_link_visual_mesh(link_name)
            else:
                mesh = self.get_link_collision_mesh(link_name)
            if mesh.volume > 0:
                mesh.apply_transform(link_pose)
                all_meshes.append(mesh)

        if to_o3d:
            return [trimesh_to_open3d(m) for m in all_meshes]
        return all_meshes


def get_urdf_mesh_collection(urdf_path, tree, mesh_scale=1.0):
    link_names = tree.get_all_link_names()
    robot_mesh = RobotMesh(urdf_path, mesh_scale=mesh_scale)
    out = {}

    for link_name in link_names:
        m = robot_mesh.get_link_collision_mesh(link_name)
        out[link_name] = m

    return out


def get_urdf_mesh(urdf_path, tree, mesh_scale=1.0):
    link_names = tree.get_all_link_names()
    robot_mesh = RobotMesh(urdf_path, mesh_scale=mesh_scale)

    out_v = []
    out_v_count = []
    out_f = []
    out_fn = []
    out_f_count = []

    for link_name in link_names:
        m = robot_mesh.get_link_collision_mesh(link_name)
        # print(link_name, m)
        
        out_v_count.append(m.vertices.shape[0])
        out_f_count.append(m.faces.shape[0])
        
        if m.vertices.shape[0] > 0:
            out_v.append(m.vertices)
            out_f.append(m.faces)
            out_fn.append(m.face_normals)

    out_v = np.concatenate(out_v)
    out_f = np.concatenate(out_f)
    out_fn = np.concatenate(out_fn)
    out_v_index = np.cumsum(np.array([0] + out_v_count))
    out_f_index = np.cumsum(np.array([0] + out_f_count))
    # print(out_v.shape, out_f.shape, out_fn.shape, out_v_index, out_f_index)

    return {"v": out_v, "f": out_f, "n": out_fn, "vi": out_v_index, "fi": out_f_index}


def get_mesh_projection_data(link_name, mesh, config):
    """ Process the mesh according to the specified rule and return a subset for contact point projection in Kinematics Finetuning.

    Args:
        link_name:  str. 
        mesh:       trimesh.Trimesh
        rule:       a rule dictionary

    Notes:
        - v1 rule: {
            "type": "v1"
            "disabled": [
                (normal_direction, threshold), 
                ...
            ] # if fn[i] is aligned with normal_direction up to threshold difference, we discard it.
        }
    """
    if link_name not in config:
        return mesh.vertices, mesh.faces, mesh.face_normals

    mesh_rule = config["movable_link"][link_name]

    if config["type"] == "v1":
        if "disabled" not in mesh_rule:
            return mesh.vertices, mesh.faces, mesh.face_normals
        
        fn = mesh.face_normals
        flags = np.ones(fn.shape[0], dtype=np.int32)

        for (n, d) in mesh_rule["disabled"]:
            flags[np.where((n * fn).sum(axis=-1) > math.cos(d))] = 0
        
        f = mesh.faces[np.where(flags == 1)]
        fn = mesh.face_normals[np.where(flags == 1)]
        return mesh.vertices, f, fn 

    return mesh.vertices, mesh.faces, mesh.face_normals


def get_urdf_mesh_for_projection(urdf_path, tree, config={}, mesh_scale=1.0):
    link_names = tree.get_all_link_names()
    robot_mesh = RobotMesh(urdf_path, mesh_scale=mesh_scale)

    out_v = []
    out_v_count = []
    out_f = []
    out_fn = []
    out_f_count = []

    for link_name in link_names:
        m = robot_mesh.get_link_collision_mesh(link_name)
        
        if m.vertices.shape[0] > 0:
            v, f, fn = get_mesh_projection_data(link_name, m, config)    
            out_v.append(v)
            out_f.append(f)
            out_fn.append(fn)
            out_v_count.append(m.vertices.shape[0])
            out_f_count.append(m.faces.shape[0])

        else:
            out_v_count.append(0)
            out_f_count.append(0)

    out_v = np.concatenate(out_v)
    out_f = np.concatenate(out_f)
    out_fn = np.concatenate(out_fn)
    out_v_index = np.cumsum(np.array([0] + out_v_count))
    out_f_index = np.cumsum(np.array([0] + out_f_count))

    return {"v": out_v, "f": out_f, "n": out_fn, "vi": out_v_index, "fi": out_f_index}



def get_urdf_mesh_decomposed(urdf_path, tree, override_link_names=None, mesh_scale=1.0):
    if override_link_names is not None:
        link_names = override_link_names
    else:
        link_names = tree.get_all_link_names()
    
    robot_mesh = RobotMesh(urdf_path, mesh_scale=mesh_scale)
    out_v = []
    out_v_count = []
    out_f = []
    out_fn = []
    out_f_count = []

    out_body_id_to_link_id = []
    out_link_body_id = {}
    mesh_id = 0

    for link_name in link_names:
        meshes = robot_mesh.get_link_collision_meshes_decomposed(link_name)
        # print(link_name, m)
        out_link_body_id[tree.get_link_id(link_name)] = []

        for m in meshes:
            out_v_count.append(m.vertices.shape[0])
            out_f_count.append(m.faces.shape[0])
            
            if m.vertices.shape[0] > 0:
                out_v.append(m.vertices)
                out_f.append(m.faces)
                out_fn.append(m.face_normals)

            out_body_id_to_link_id.append(tree.get_link_id(link_name))
            out_link_body_id[tree.get_link_id(link_name)].append(mesh_id)
            # print(link_name, mesh_id)

            mesh_id += 1

    out_v = np.concatenate(out_v)
    out_f = np.concatenate(out_f)
    out_fn = np.concatenate(out_fn)
    out_v_index = np.cumsum(np.array([0] + out_v_count))
    out_f_index = np.cumsum(np.array([0] + out_f_count))
    # print(out_v.shape, out_f.shape, out_fn.shape, out_v_index, out_f_index)

    return {
        "v": out_v, 
        "f": out_f, 
        "n": out_fn, 
        "vi": out_v_index, 
        "fi": out_f_index, 
        "body_id_to_link_id": out_body_id_to_link_id, 
        "link_body_id": out_link_body_id
    }


