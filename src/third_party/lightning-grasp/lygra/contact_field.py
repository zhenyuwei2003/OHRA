# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.kinematics import batch_fk
from lygra.utils.geom_utils import get_tangent_plane
from lygra.contact.accel.lbvh_s2bundle import LBVHS2ContactFieldAccelStructure
from lygra.mesh_analyzer import get_robot_link_mesh_group
from lygra.utils.vis_utils import trimesh_to_open3d, make_arrow_lines, show_visualization
from lygra.utils.save_utils import save_json, save_pickle, load_json, load_pickle
from lygra.utils.transform_utils import batch_apply_transform_np, batch_apply_rotation_np

from tqdm import tqdm
from pathlib import Path
import numpy as np 
import torch
import math 



class ContactField:
    def __init__(self):
        """
            This class organizes necessary data structure for building and sampling from Contact Field
        """

        # These are movable contact patches.
        self.all_patch_parent_link_names = []       # len(N)
        self.all_patch_parent_link_idx = []         # len(N)
        self.all_patch_keyvectors = []              # len(N): a list of [N_keyvector, 6] vectors.
        self.all_patch_keyvector_ids = []           # len(N)
        self.all_patch_keyvector_num = []           # len(N)
        self.all_patch_keyvector_offset = []        # len(N+1)
        self.all_patch_angle_range = []
        self.all_patch_centroid = [] 
        self.all_patch_centroid_torch = None 
        self.all_patch_keyvectors_torch = None 

        self.all_static_patch_keyvectors = []

        self.all_contact_link_names = []
        self.all_contact_link_contact_field_idxes = []

        self.field_vec_origin = []
        self.field_vec_direction = []

        self.accel_structure = None

    def sample_contact_ids(
        self, 
        interaction_matrix,
        interaction_matrix_hand_point_idx,
        target_batch_outer_ids, 
        target_contact_link_ids, 
        target_contact_point_idx
    ):
        """
        Args:
            interaction_matrix:                      [batch_outer, n_contact_field, n_object_point]
            interaction_matrix_hand_point_idx:       [batch_outer, n_contact_field, n_object_point]  (the nearest contact point id.)
            target_batch_outer_ids:                  [n_success]
            target_contact_link_ids:                 [n_success, n_contact]
            target_contact_point_idx:                [n_success, n_contact]

        Returns:
            contact_field_ids:                       [n_success, n_contact]
            local_contact_point_ids:                 [n_success, n_contact]

        Note:
            for each (i, j) in target_contact_point_idx:
                active_contact_field_ids = interaction_matrix[i, j]
                contact_ids[i, j] = random(active_ids)
        """

        mask = interaction_matrix.permute(0, 2, 1) # [batch_outer, n_object_point, n_contact_field]
        hand_point_ids = interaction_matrix_hand_point_idx.permute(0, 2, 1)

        target_batch_outer_ids = target_batch_outer_ids.view(-1, 1).expand(
            -1, target_contact_point_idx.size(-1)
        )  # [n_success, n_contact]

        mask = mask[target_batch_outer_ids, target_contact_point_idx].bool()              # [n_success, n_contact, n_contact_field]
        hand_point_ids = hand_point_ids[target_batch_outer_ids, target_contact_point_idx] # [n_success, n_contact, n_contact_field]
        link_mask = self.contact_field_selection_tensor[target_contact_link_ids]          # [n_success, n_contact, n_contact_field]
        mask = torch.logical_and(mask, link_mask)
        assert mask.any(dim=-1).all(), "Mask Error."

        rand = torch.rand_like(mask.float())
        rand_masked = rand.masked_fill(~mask, float('-inf'))
        contact_field_ids = rand_masked.argmax(dim=-1)          
        local_contact_point_ids = torch.gather(hand_point_ids, dim=-1, index=contact_field_ids.unsqueeze(-1)).squeeze(-1)
        assert (local_contact_point_ids >= 0).all()

        return contact_field_ids, local_contact_point_ids

    def reduce_link_interaction(self, interaction_matrix):
        """
        
        Args:
            interaction_matrix: [B, N_field, N_point]

        Returns:
            link_interaction_matrix: [B, N_link, N_point]
        """

        result = []
        for idxes in self.all_contact_link_contact_field_idxes:
            result.append(interaction_matrix[:, idxes, :].bool().any(dim=1).int())
        return torch.stack(result, dim=1)

    def register_keyvector_static(self, link_name, keyvectors, centroid):
        '''
            Static keyvector is for the palm...
        
        Args:
            keyvectors: [n, 6]
            centroid:   [1, 6]
        '''
        # we assume this is already in the hand base frame.
        self.all_static_patch_keyvectors.append(keyvectors)
        return 

    def register_keyvector(self, link_name, keyvectors, centroid, angle_range):
        if len(keyvectors) == 0:
            print("[Warning] Skipping empty key vector")
            return 

        self.all_patch_parent_link_names.append(link_name)
        self.all_patch_keyvectors.append(keyvectors)
        self.all_patch_keyvector_num.append(len(keyvectors))
        self.all_patch_centroid.append(centroid.squeeze())
        self.all_patch_angle_range.append(angle_range)
        return 

    def get_all_contact_link_names(self):
        return self.all_contact_link_names

    def get_all_parent_link_names(self):
        return self.all_patch_parent_link_names


    def sample_spatial_contact(self, batch_size, sampling_args):
        if sampling_args["strategy"] == 'canonical':
            bmin = sampling_args.get('bmin', None)
            bmax = sampling_args.get('bmax', None)

            if bmin is not None:
                center = (bmax + bmin) / 2
                p = np.random.uniform(bmin, bmax, size=(batch_size, 3))
                d = np.random.randn(batch_size, 3)
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                return np.concatenate([p, d], axis=-1)

        # Default fallback option
        return self.accel_structure.sample_spatial_contact(batch_size)


    def sample_static_contact_geometry(self, batch_size):
        '''
        Args:
            int. number of samples.
        '''
        if len(self.all_static_patch_keyvectors) == 0:
            return None

        return self.all_static_patch_keyvectors[np.random.randint(0, len(self.all_static_patch_keyvectors), batch_size)]


    def sample_contact_geometry(self, index_tensor, local_id_tensor=None):
        '''
        Args:
            index_tensor: [...], a int tensor. It specifies the used contact patch.

        Returns:
            contact_pos:    [..., 3]. The contact position expressed in the patch's parent link's frame.
            contact_normal: [..., 3]. The contact position expressed in the patch's parent link's frame.
        '''
        if local_id_tensor is not None:
            assert index_tensor.shape == local_id_tensor.shape 

            if self.all_patch_keyvectors_torch is None:
                self.all_patch_keyvectors_torch = torch.from_numpy(np.array(self.all_patch_keyvectors)).to(index_tensor.device)

            all_patch_origin = self.all_patch_keyvectors_torch[:, :, :3].to(index_tensor.device)             # [N, K, 3]
            all_patch_direction = self.all_patch_keyvectors_torch[:, :, 3:].to(index_tensor.device)                                  # [N, K, 3]
        
            contact_pos = all_patch_origin[index_tensor, local_id_tensor]
            contact_normal = all_patch_direction[index_tensor, local_id_tensor]

            return contact_pos, contact_normal

        else:
            if self.all_patch_centroid_torch is None:
                self.all_patch_centroid_torch = torch.from_numpy(np.array(self.all_patch_centroid)).to(index_tensor.device)
            
            all_patch_origin = self.all_patch_centroid_torch[:, :3].to(index_tensor.device)             # [N, 3]
            all_patch_direction = self.all_patch_centroid_torch[:, 3:].to(index_tensor.device)          # [N, 3]
        
            contact_pos = all_patch_origin[index_tensor]
            contact_normal = all_patch_direction[index_tensor]

            return contact_pos, contact_normal


    def generate_contact_field(self, tree, n_iter=80000, device='cuda'):
        joint_limit_lower, joint_limit_upper = tree.get_active_joint_limit()
        joint_limit_lower = torch.from_numpy(joint_limit_lower).to(device)
        joint_limit_upper = torch.from_numpy(joint_limit_upper).to(device)        
        q = torch.rand(n_iter, joint_limit_lower.shape[-1]).to(device) * (joint_limit_upper - joint_limit_lower) + joint_limit_lower

        # we need to get the pose of the 
        link_name_set = list(set(self.all_patch_parent_link_names))
        link_idx_set = {name: tree.get_link_id(name) for name in link_name_set}
        link_idx_list = link_idx_set.values()

        all_index = []
        self.all_patch_parent_link_idx = []

        for i in range(len(self.all_patch_parent_link_names)):
            name = self.all_patch_parent_link_names[i]
            idx = link_idx_set[name]
            self.all_patch_parent_link_idx.append(idx)
            for _ in range(self.all_patch_keyvector_num[i]):
                all_index.append(idx)
        
        all_index = np.array(all_index)
        transform = np.tile(np.eye(4)[None], (len(all_index), 1, 1))

        combined_keyvectors = np.concatenate(self.all_patch_keyvectors)
        combined_origin = combined_keyvectors[:, :3]
        combined_normal = combined_keyvectors[:, 3:]

        result_origins = []
        result_directions = []

        # we will need to fill these transforms.
        # In each loop, we need a [N_ray, 4, 4] matrix to transform the ray, and we record the resulting ray field.
        link_poses = batch_fk(tree, q)["link"].detach().cpu().numpy()

        for i in tqdm(range(n_iter), desc="Sampling Contact Field"):
            for j, link_idx in enumerate(link_idx_list):
                transform[np.where(all_index == link_idx)] = link_poses[i, link_idx]

            # then we apply transform over the rays.
            ray_origins = batch_apply_transform_np(combined_origin, transform)
            ray_directions = batch_apply_rotation_np(combined_normal, transform)
            
            result_origins.append(ray_origins)
            result_directions.append(ray_directions)
 
        result_origin = np.stack(result_origins, axis=1)        # [N_ALL_KEYVECTORS, N_sample, 3]
        result_direction = np.stack(result_directions, axis=1)  # [N_ALL_KEYVECTORS, N_sample, 3]

        self.all_patch_keyvector_offset = np.cumsum([0] + self.all_patch_keyvector_num)
        self.field_vec_origin = []
        self.field_vec_direction = []
        self.field_vec_id = []

        for i in range(len(self.all_patch_keyvector_offset) - 1):
            begin = self.all_patch_keyvector_offset[i]
            end = self.all_patch_keyvector_offset[i+1]
          
            origin = result_origin[begin:end].reshape(-1, 3)
            direction = result_direction[begin:end].reshape(-1, 3)
            
            n = origin.shape[0] // (end - begin)
            field_vec_ids = np.zeros(origin.shape[0], dtype=np.int32)
            for j in range(end - begin):
                field_vec_ids[j * n : (j + 1) : n] = j

            self.field_vec_origin.append(origin)
            self.field_vec_direction.append(direction)
            self.field_vec_id.append(field_vec_ids)

        if len(self.all_static_patch_keyvectors) > 0:
            self.all_static_patch_keyvectors = np.concatenate(self.all_static_patch_keyvectors) # [n, 6]

        self.all_contact_link_names = list(set(self.all_patch_parent_link_names))

        self.contact_field_selection_tensor = torch.zeros(len(self.all_contact_link_names), len(self.all_patch_parent_link_names)).bool()
        for contact_link_id, parent_link in enumerate(self.all_contact_link_names):
            idxes = []
            for i, n in enumerate(self.all_patch_parent_link_names):
                if parent_link == n:
                    idxes.append(i)
            
            self.contact_field_selection_tensor[contact_link_id, idxes] = True

            self.all_contact_link_contact_field_idxes.append(idxes)

        self.contact_field_selection_tensor = self.contact_field_selection_tensor.cuda()

    def generate_acceleration_structure(self, method='lbvhs2', **kwargs):
        orgs = self.field_vec_origin
        dirs = self.field_vec_direction
        angle_range = np.array(self.all_patch_angle_range)

        if method == 'lbvhs2':
            structure = LBVHS2ContactFieldAccelStructure(orgs, dirs, self.field_vec_id, angle_range, **kwargs)

        else:
            assert False, "Approximation method not supported."
        
        self.accel_structure = structure
        return structure 

    # TODO(): We will need these. No need to generate contact field every time.
    def save(self, folder):
        """ Save the contact field.

        Folder strucuture:
        - config.json
        - contact_field.pkl
        - accel_weights (if any, handled by acceleration structure)
        """
        if not isinstance(folder, Path):
            folder = Path(folder)

        os.makedirs(folder, exist_ok=True)
        # Save config
        config = {
            "accel": self.accel_structure.type
        }
        save_json(config, path / "config.json")

        # Save data
        contact_field_data = {
            "all_patch_parent_link_names": self.all_patch_parent_link_names,
            "all_patch_parent_link_idx":   self.all_patch_parent_link_idx,
            "all_patch_keyvectors":        self.all_patch_keyvectors,
            "all_patch_keyvector_num":     self.all_patch_keyvector_num,
            "all_patch_keyvector_offset":  self.all_patch_keyvector_offset,
            "all_patch_angle_range":       self.all_patch_angle_range,
            "all_patch_centroid":          self.all_patch_centroid,
            "all_static_patch_keyvectors": self.all_static_patch_keyvectors,
        }
        save_pickle(contact_field_data, path / "contact_field.pkl")

        self.accel_structure.save(folder)

    # TODO()
    def load(self, folder):
        if not isinstance(folder, Path):
            folder = Path(folder)

        config = load_json(path / "config.json")
        contact_field_data = load_pickle(path / "contact_field.pkl")

        self.all_patch_parent_link_names = data["all_patch_parent_link_names"]
        self.all_patch_parent_link_idx   = data["all_patch_parent_link_idx"]
        self.all_patch_keyvectors        = data["all_patch_keyvectors"]
        self.all_patch_keyvector_num     = data["all_patch_keyvector_num"]
        self.all_patch_keyvector_offset  = data["all_patch_keyvector_offset"]
        self.all_patch_angle_range       = data["all_patch_angle_range"]
        self.all_patch_centroid          = data["all_patch_centroid"]
        self.all_static_patch_keyvectors = data["all_static_patch_keyvectors"]
        return


def has_aligned(query, vecs, threshold=3.1415926 * 0.49999):
    return np.any((query * vecs).sum(axis=-1) > math.cos(threshold))


def get_default_rule():
    rule = {
        "movable": {
            "disable_normals": [] 
        },
        "use_link": [],
        "static_link": [],
        "static": {
            "allow_normals": []
        }
    }
    return rule


def build_contact_field(
    urdf_path, 
    rule, 
    mesh_scale=1.0,
    patch_pos_dist_tol=0.0075, 
    patch_rot_dist_tol=3.1415926 * 0.15
):
    """
        Rule format:
        {
            "type": "v1",
            "movable_link": {
                link_name: rule
            }
            "static_link": {
                link_name: rule
            }
        }
    """
    contact_field = ContactField()
    link_mesh_group = get_robot_link_mesh_group(
        urdf_path, 
        mesh_scale=mesh_scale, 
        patch_pos_dist_tol=patch_pos_dist_tol,
        patch_rot_dist_tol=patch_rot_dist_tol
    )

    for link_name, mesh_group in link_mesh_group.items():
        n_group = len(mesh_group["group_vectors"])

        if link_name not in rule["movable_link"] and link_name not in rule["static_link"]:
            continue

        for i in range(n_group):
            keyvector = mesh_group["group_vectors"][i]
            centroid = mesh_group["group_centroid"][i]
            angle_range = mesh_group["group_angle_range"][i]
            if link_name in rule["static_link"]:
                # static link: e.g. palm
                for allow_static_normals, threshold in rule["static_link"][link_name]["allowed_normal"]:
                    if has_aligned(centroid[:, 3:], allow_static_normals, threshold=threshold):
                        contact_field.register_keyvector_static(link_name, keyvector, centroid)
                        break
            else:
                # movable link.
                passed = True

                for disabled_normals, threshold in rule["movable_link"][link_name]["disabled_normal"]:
                    if has_aligned(keyvector[:, 3:], disabled_normals, threshold=threshold):
                        passed = False
                        break
    
                if passed:
                    contact_field.register_keyvector(link_name, keyvector, centroid, angle_range)

    return contact_field

