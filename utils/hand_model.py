import os
import sys
import json
import time
import torch
import trimesh
import fpsample
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


def parse_urdf_to_trimesh(urdf_type, urdf_file, link_names=None, use_collision=False):
    root = ET.parse(urdf_file).getroot() if urdf_type == 'path' else ET.fromstring(urdf_file)
    trimesh_dict = {}

    for link in root.findall('link'):
        link_name = link.get('name')
        if link_names is not None and link_name not in link_names:
            continue

        meshes = []
        for obj in link.findall(f".//{'collision' if use_collision else 'visual'}"):
            origin = obj.find('origin')
            if origin is None:
                xyz = np.zeros(3)
                rpy = np.zeros(3)
            else:
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')

            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = R.from_euler('xyz', rpy).as_matrix()

            geometry = obj.find('geometry')
            mesh = None

            if geometry.find('box') is not None:
                box = geometry.find('box')
                size = np.fromstring(box.get('size'), sep=' ')
                mesh = trimesh.creation.box(extents=size)

            elif geometry.find('cylinder') is not None:
                cylinder = geometry.find('cylinder')
                radius = float(cylinder.get('radius'))
                height = float(cylinder.get('length'))
                mesh = trimesh.creation.cylinder(radius=radius, height=height)

            elif geometry.find('sphere') is not None:
                sphere = geometry.find('sphere')
                radius = float(sphere.get('radius'))
                mesh = trimesh.creation.icosphere(radius=radius)

            elif geometry.find('mesh') is not None:
                mesh_elem = geometry.find('mesh')
                filename = mesh_elem.get('filename')
                if filename:
                    mesh_path = os.path.join(os.path.dirname(urdf_file), filename)
                    try:
                        mesh = trimesh.load(mesh_path)
                        scale = mesh_elem.get('scale')
                        if scale:
                            scale = np.fromstring(scale, sep=' ')
                            scale_transform = np.eye(4)
                            scale_transform[0, 0] = scale[0]
                            scale_transform[1, 1] = scale[1]
                            scale_transform[2, 2] = scale[2]
                            mesh.apply_transform(scale_transform)
                    except Exception as e:
                        print(f"Failed to load mesh {filename} for link {link_name}: {e}")
                        mesh = None

            if mesh is not None:
                mesh.apply_transform(transform)
                meshes.append(mesh)

        if meshes:
            combined_mesh = trimesh.util.concatenate(meshes)
            trimesh_dict[link_name] = combined_mesh

    return trimesh_dict


class HandModel:
    def __init__(self, robot_name, urdf_type, urdf_file, is_canonical, use_collision=False, device=torch.device('cpu')):
        self.robot_name = robot_name
        self.urdf_type = urdf_type  # 'path' or 'string'
        self.urdf_file = urdf_file
        self.is_canonical = is_canonical
        self.use_collision = use_collision
        self.device = device

        self.pk_chain = pk.build_chain_from_urdf(self.urdf_file if urdf_type == 'string' else open(self.urdf_file, 'rb').read()).to(dtype=torch.float32, device=device)
        self.meta_info = json.load(open(os.path.join(ROOT_DIR, f"assets/meta_infos/{self.robot_name}.json")))
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.trimeshes = parse_urdf_to_trimesh(self.urdf_type, self.urdf_file, use_collision=use_collision)
        self.joint_orders = [joint.name for joint in self.pk_chain.get_joints()]
        self.link2joint_map = self.get_link2joint_map()  # child link name -> joint name
        self.joint2link_map = {v: k for k, v in self.link2joint_map.items()}  # joint name -> child link name

        self.frame_status = None

    def get_link2joint_map(self):
        link2joint_map = {}
        root = ET.parse(self.urdf_file).getroot() if self.urdf_type == 'path' else ET.fromstring(self.urdf_file)
        for joint in root.findall('joint'):
            if joint.get('type') == 'revolute':
                link2joint_map[joint.find('child').get('link')] = joint.get('name')
        return link2joint_map

    def get_joint_limits(self):
        lower, upper = self.pk_chain.get_joint_limits()
        lower = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper = torch.tensor(upper, dtype=torch.float32, device=self.device)
        return lower, upper

    def get_initial_q(self):
        q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        lower, upper = self.get_joint_limits()
        for i in range(self.dof):
            q[i] = min(max(0, lower[i]), upper[i])
        return q
    
    def get_canonical_ordered_q(self, q, canonical_order=None):
        assert self.is_canonical, "Conversion is required when not using a canonical hand model."
        assert len(q) == self.meta_info["dof"], f"Expected {self.meta_info['dof']} DoF for {self.robot_name}, got {len(q)}"
        canonical_order = self.meta_info["canonical_order"] if canonical_order is None else canonical_order

        q_canonical = [np.sign(x) * q[abs(x) - 1] for x in canonical_order]
        if self.robot_name == 'barrett_hand':  # Barrett's last finger joints are not parallel at initial position, adjust them here
            q_canonical[4] -= 0.785
            q_canonical[8] += 0.785
            q_canonical[12] += 0.785
        elif self.robot_name == "allegro_hand_left":
            q_canonical[5] = 0.088
            q_canonical[18] = -0.088
        elif self.robot_name == "allegro_hand_right":
            q_canonical[5] = -0.088
            q_canonical[18] = 0.088
        return q_canonical

    def get_original_ordered_q(self, q_canonical):
        assert not self.is_canonical, "Conversion is not required when using a canonical hand model."
        assert len(q_canonical) == 22, f"Expected 22 DoF for canonical_assets urdf, got {len(q_canonical)}"

        q = [0.0] * self.meta_info["dof"]
        for canonical_idx, x in enumerate(self.meta_info["canonical_order"]):
            if x != 0:
                original_idx = abs(x) - 1
                q[original_idx] = np.sign(x) * q_canonical[canonical_idx]
        if self.robot_name == 'barrett_hand':
            q[2] += 0.785
            q[5] += 0.785
            q[7] += 0.785
        return q

    def get_trimeshes_q(self, q):
        self.frame_status = self.pk_chain.forward_kinematics(q)

        scene = trimesh.Scene()
        link_trimeshes = {}
        for link_name in self.trimeshes:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            link_trimesh = self.trimeshes[link_name].copy().apply_transform(mesh_transform_matrix)
            link_trimeshes[link_name] = link_trimesh
            scene.add_geometry(link_trimesh)
        
        vertices, faces = [], []
        vertex_offset = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                vertex_offset += len(geom.vertices)
        vertices = np.vstack(vertices)
        faces = np.vstack(faces)

        return {
            'link': link_trimeshes,
            'all': trimesh.Trimesh(vertices=vertices, faces=faces)
        }

    def get_transformed_pc(self, q, num_points=512):
        frame_status = self.pk_chain.forward_kinematics(q)

        point_clouds = []
        for link_name, link_trimesh in self.trimeshes.items():
            mesh_transform = frame_status[link_name].get_matrix()[0].cpu().numpy()
            link_trimesh_transformed = link_trimesh.copy().apply_transform(mesh_transform)
            point_clouds.append(link_trimesh_transformed.sample(num_points))
        point_clouds = np.vstack(point_clouds)
        sample_idx = fpsample.bucket_fps_kdtree_sampling(point_clouds, num_points)
        point_clouds = point_clouds[sample_idx]

        return torch.from_numpy(point_clouds).to(dtype=torch.float32, device=self.device)

    def sample_random_config(self, num_samples):
        lower, upper = self.pk_chain.get_joint_limits()
        configs = np.random.uniform(lower, upper, size=(num_samples, self.dof)).astype(np.float32)
        if self.robot_name == "shadow_hand" and not self.is_canonical:
            configs[:, :2] = 0  # set WRJ1 and WRJ2 to 0
        return configs
