# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import trimesh
from urdfpy import URDF
from pathlib import Path 
import itertools
import math
import sys 
import torch
from collections import defaultdict
import time 
from lygra.utils.robot_visualizer import RobotMesh
from lygra.utils.transform_utils import batch_axis_angle, batch_translation
from tqdm import tqdm 


def pair_list_to_dict(pairs):
    """
    Converts a list of (i, j) pairs into a dict where each i maps to a list of j's.

    Args:
        pairs (List[Tuple[Any, Any]]): List of (i, j) pairs

    Returns:
        Dict[i, List[j]]: Dictionary mapping i to list of corresponding j's
    """
    result = defaultdict(list)
    for i, j in pairs:
        result[i].append(j)
    return dict(result)


class KinematicsTree:
    """
    This is an auxiliary data structure for forward kinematics and jacobian calculation.
    
    TODO: This version does not support passive joints (mimic joint)
          I need to finish this in a future update.
    """
    
    def __init__(self):
        self.objects = []
        self.edges = {}
        self.edges_rev = {}

        self.object_name_to_id = {}
        self.object_id = 0
        
        self.links = [] # strs, link names.
        self.link_name_to_id = {}
        self.adjacent_link_pairs = []

        self.joints = []
        self.joint_parent_link_ids = []                                           # input
        self.joint_child_link_ids = []                                            # input
        self.joint_types = []                                                     # input
        self.joint_axes = []                                                      # input
        self.joint_origins = []                                                   # input
        self.joint_limit_lowers = []
        self.joint_limit_uppers = []
        self.active_joints = []

        self.active_joint_lowers = None 
        self.active_joint_uppers = None
        self.base_link = None
        return 

    @staticmethod
    def build_from_urdf(urdf_path, active_joint_names=None):
        urdf = URDF.load(urdf_path)
        tree = KinematicsTree()
        for l in urdf.links:
            tree.add_link(l.name)

        for j in urdf.joints:
            tree.add_joint(
                j.name, 
                parent_name=j.parent, 
                child_name=j.child, 
                joint_type=j.joint_type,
                axis=j.axis, 
                limit=j.limit,
                origin=j.origin
            )
            tree.add_edge(j.parent, j.name)
            tree.add_edge(j.name, j.child)

        if active_joint_names is None:
            active_joints = urdf._actuated_joints                       # this will be the indexing order.
            active_joint_names = [j.name for j in active_joints] 

        tree.set_active_joints(active_joint_names)
        return tree

    def get_base_link(self):
        if self.base_link is None:
            for l in self.links:
                if len(self.edges_rev[l]) == 0:
                    self.base_link = l 
                    # print("Kinematics Tree Base Link (Root):", self.base_link)
                    break

        return self.base_link 

    def n_link(self):
        return len(self.links)

    def n_dof(self):
        return len(self.active_joints)

    def add_link(self, name):
        self.links.append(name)
        self.add_item(name)
        return 

    def get_link_id(self, name):
        if name not in self.link_name_to_id:
            self.link_name_to_id[name] = self.links.index(name)
        
        return self.link_name_to_id[name]

    def get_self_collision_check_link_pairs(self, link_body_id, whitelist_pairs=[], whitelist_link=[]):
        n_link = self.n_link()

        whitelist_id =  [(self.get_link_id(i), self.get_link_id(j)) for (i, j) in whitelist_pairs] 
        whitelist_id += [(self.get_link_id(j), self.get_link_id(i)) for (i, j) in whitelist_pairs] 
        
        filter_pair_list = whitelist_id + self.adjacent_link_pairs
        filter_link_list = [self.get_link_id(i) for i in whitelist_link]

        result = []
        for i in range(n_link):
            for j in range(i + 1, n_link):
                id0 = self.get_link_id(self.links[i])
                id1 = self.get_link_id(self.links[j])

                if (id0, id1) in filter_pair_list or id0 in filter_link_list or id1 in filter_link_list:
                    continue

                body0 = link_body_id[id0]
                body1 = link_body_id[id1]

                for b0 in body0:
                    for b1 in body1:
                        result.append([b0, b1])
        
        return np.array(result, dtype=np.int32) 

    def add_joint(self, name, parent_name, child_name, joint_type, axis, limit, origin):
        parent_link_id = self.get_link_id(parent_name)
        child_link_id = self.get_link_id(child_name)
        self.joints.append(name)
        self.joint_parent_link_ids.append(parent_link_id)
        self.joint_child_link_ids.append(child_link_id)
        self.joint_types.append(joint_type)
        self.joint_axes.append(axis)
        self.joint_origins.append(origin)
        if limit is not None:
            self.joint_limit_lowers.append(limit.lower)
            self.joint_limit_uppers.append(limit.upper)
        else:
            self.joint_limit_lowers.append(0.0)
            self.joint_limit_uppers.append(0.0)

        self.add_item(name)
        self.adjacent_link_pairs.append((parent_link_id, child_link_id))
        self.adjacent_link_pairs.append((child_link_id, parent_link_id))
        return 

    def add_item(self, name):
        object_id = self.object_id
        self.objects.append(name)
        self.object_name_to_id[name] = object_id
        self.edges[name] = []
        self.edges_rev[name] = []
        self.object_id += 1
        
        return object_id

    def get_id(self, name):
        return self.object_name_to_id[name]

    def add_edge(self, u_name, v_name):
        self.edges[u_name].append(v_name)
        self.edges_rev[v_name].append(u_name)
        return 

    def joint_topological_sort(self, root):
        visited = []
        stack = [root]

        while len(stack) > 0:
            v = stack.pop()
            if v in visited:
                continue # however, a tree does not require this.

            visited.append(v)
            for child in self.edges[v]:
                stack.append(child)

        visited_joint = [o for o in visited if o in self.joints]
        return visited_joint

    def get_joint_id(self, joint):
        if isinstance(joint, list):
            return [self.get_joint_id(j) for j in joint]
        else:
            return self.joints.index(joint)

    def set_active_joints(self, active_joints):
        self.active_joints = active_joints 
        return

    def get_active_joint_limit(self):
        if self.active_joint_uppers is None:
            all_jids = [self.get_joint_id_from_active_joint_id(i) for i in range(len(self.active_joints))]
            self.active_joint_lowers = np.array([self.joint_limit_lowers[jid] for jid in all_jids]).astype(np.float32)
            self.active_joint_uppers = np.array([self.joint_limit_uppers[jid] for jid in all_jids]).astype(np.float32)

        return self.active_joint_lowers, self.active_joint_uppers

    def get_dependency_sets(self, observed_links):
        # Remove the observed links from the graph, and return the connected components.
        link_to_remove = []

        for link in observed_links:            
            # do a dfs.
            visited = []
            stack = [link]

            while len(stack) > 0:
                v = stack.pop()

                link_to_remove.append(v)
                visited.append(v)

                for joint in self.edges[v]:
                    if joint not in self.active_joints:
                        # this is a fixed link, we should remove the child as well.
                        for child in self.edges[joint]:
                            stack.append(child)

        dependency_sets = []
        visited_links = {l: False for l in self.links}

        # Find connected components.
        for link in self.links:
            if link in link_to_remove:
                continue

            if visited_links[link]:
                continue 

            dependency_set = []
            visited = []
            stack = [link]

            while len(stack) > 0:
                v = stack.pop()
                dependency_set.append(v)
                visited_links[v] = True 
                visited.append(v)

                for joint in self.edges[v]:              
                    for child in self.edges[joint]:
                        if child not in link_to_remove + visited:
                            stack.append(child)

                for joint in self.edges_rev[v]:              
                    for child in self.edges_rev[joint]:
                        if child not in link_to_remove + visited:
                            stack.append(child)

            dependency_sets.append(dependency_set)
        
        return dependency_sets

    def get_chain_matrix(self):
        """

        Returns:
            [n_link, n_dof] a binary matrix.
        """
        out = np.zeros((self.n_link(), self.n_dof())).astype(np.float32)
        jacobian_pairs = self.get_jacobian_pairs(to_id=True)

        for joint_id, link_id in jacobian_pairs:
            out[link_id, self.get_active_joint_id_from_joint_id(joint_id)] = 1.0
        return out

    def get_all_link_names(self):
        return self.links 

    def get_joint_id_from_active_joint_id(self, active_joint_id):
        """ given an active joint id, returns its joint id"""
        if self.active_joints[active_joint_id] in self.joints:
            return self.joints.index(self.active_joints[active_joint_id])
        return -1 

    def get_active_joint_id_from_joint_id(self, joint_id):
        """ given a joint id, returns its active_id in the [n_dof] qpos tensor"""
        if self.joints[joint_id] in self.active_joints:
            return self.active_joints.index(self.joints[joint_id])
        return -1 

    def get_all_joint_types(self):
        return self.joint_types 

    def get_all_joint_axes(self):
        return self.joint_axes

    def get_all_joint_origins(self):
        return self.joint_origins
    
    def get_all_joint_parent_ids(self):
        return self.joint_parent_link_ids
    
    def get_all_joint_child_ids(self):
        return self.joint_child_link_ids

    def get_resting_q(self):
        return (np.array(self.joint_limit_lowers) + np.array(self.joint_limit_uppers)).astype(np.float32) / 2

    def get_jacobian_pairs(self, to_id=True):
        """
        Returns:
            out: a list of (joint_id, link_id). 
        
        Note: 
            1. If (i, j) is in result, then link j can be affected by joint i. 
            2. The returned id is joint_id, not active_joint_id.
        """
        root = self.get_base_link()
        # Again, do dfs!  
        visited = []
        stack = [root]

        chain_joints = []
        chain_pop_signals = []
        
        result = []

        while len(stack) > 0:
            v = stack.pop()

            if len(chain_pop_signals) > 0:
                while chain_pop_signals[-1] == v:
                    chain_pop_signals.pop()
                    chain_joints.pop()
                    if len(chain_pop_signals) == 0:
                        break 

            if v in visited:
                continue # however, a tree does not require this.

            visited.append(v)

            # Node actions.
            # Action 1: Add a joint to the tracked chain!
            if v in self.active_joints:
                chain_joints.append(v)
                if len(stack) > 0:
                    # Remember when to pop this out!
                    chain_pop_signals.append(stack[-1])
            
             # Action 2: Link. All the tracked joint can affect the currently visited link. 
            if v in self.links:
                for j in chain_joints:
                    result.append((j, child))

            for child in self.edges[v]:
                stack.append(child)

        if to_id:
            return [(self.joints.index(i), self.links.index(j)) for (i, j) in result]
        return result


def build_kinematics_tree(
    urdf_path='../../geort_update/assets/allegro_right/allegro_hand_right.urdf', 
    active_joint_names=None
):
    return KinematicsTree.build_from_urdf(urdf_path, active_joint_names)
   

def batch_fk(
    tree, 
    joint_q, 
    ret_info=False, 
    all_joint_result_buffer=None, 
    all_link_result_buffer=None, 
    all_joint_origins=None, 
    all_joint_axis=None
):
    batch_size = joint_q.size(0)
    joint_axes = np.array(tree.get_all_joint_axes()).astype(np.float32)
    joint_origins = np.array(tree.get_all_joint_origins()).astype(np.float32)
    sorted_joints = tree.get_joint_id(tree.joint_topological_sort(tree.get_base_link()))

    n_dof = tree.n_dof()
    n_link = tree.n_link()

    joint_parent_link_ids = tree.get_all_joint_parent_ids()
    joint_child_link_ids = tree.get_all_joint_child_ids()
    joint_types = tree.get_all_joint_types() 

    if all_joint_origins is None:
        all_joint_origins = torch.from_numpy(joint_origins).to(joint_q.device).unsqueeze(0).expand(batch_size, -1, -1, -1)  # input
    if all_joint_axis is None:
        all_joint_axis = torch.from_numpy(joint_axes).to(joint_q.device).unsqueeze(0).expand(batch_size, -1, -1)            # input

    if all_joint_result_buffer is not None:
        all_joint_result = all_joint_result_buffer
    else:
        all_joint_result = torch.zeros([batch_size, n_dof, 4, 4]).to(joint_q.device)     

    if all_link_result_buffer is not None:
        all_link_result = all_link_result_buffer
    else:
        all_link_result = torch.zeros([batch_size, n_link, 4, 4]).to(joint_q.device)   
        all_link_result[:, tree.get_link_id(tree.get_base_link()), :, :] = torch.eye(4).expand(batch_size, 4, 4).to(joint_q.device)

    for joint_id in sorted_joints:
        parent_link_id = joint_parent_link_ids[joint_id]
        child_link_id = joint_child_link_ids[joint_id]
        joint_type = joint_types[joint_id]

        parent_link_transform = all_link_result[:, parent_link_id]  # T_base_parent
        
        # calculate joint pose
        joint_pose = torch.bmm(parent_link_transform, all_joint_origins[:, joint_id])

        active_joint_id = tree.get_active_joint_id_from_joint_id(joint_id)
        if active_joint_id >= 0:
            all_joint_result[:, active_joint_id] = joint_pose 

        # calculate child pose.
        if joint_type == 'revolute':
            transform = batch_axis_angle(joint_q[:, active_joint_id], all_joint_axis[:, joint_id]) 
            child_link_transform = torch.bmm(joint_pose, transform)

        elif joint_type == 'prismatic':
            transform = batch_translation(joint_q[:, active_joint_id], all_joint_axis[:, joint_id]) # should be [B, 4, 4]
            child_link_transform = torch.bmm(joint_pose, transform)
        
        else:
            child_link_transform = joint_pose 
        
        all_link_result[:, child_link_id] = child_link_transform

    result = {"link": all_link_result, "joint": all_joint_result}
    if ret_info:
        result["axes"] = all_joint_axis
    return result


def batch_jacobian(
    tree, 
    joint_q, 
    fk_joint_buffer=None,
    fk_link_buffer=None, 
    out=None, 
    joint_types=None, 
    jacobian_pairs=None,
    ret_fk=False
):
    batch_size = joint_q.size(0)
    if joint_types is None:
        joint_types = tree.get_all_joint_types() 

    if jacobian_pairs is None:
        jacobian_pairs = tree.get_jacobian_pairs(to_id=True)
 
    result = batch_fk(
        tree, 
        joint_q, 
        ret_info=True, 
        all_joint_result_buffer=fk_joint_buffer, 
        all_link_result_buffer=fk_link_buffer
    )

    all_link_result = result["link"] 
    all_joint_result = result["joint"]
    all_joint_axis = result["axes"]

    if out is None:
        # Transposed version to maximize the write throughput
        out = torch.zeros(tree.n_dof(), tree.n_link() * 6, batch_size).to(joint_q.device)
    
    # note: using cuda stream for this will be slower.
    all_link_result_cont = all_link_result.transpose(0, 1).contiguous()
    all_joint_result_cont = all_joint_result.transpose(0, 1).contiguous()
    
    for i, (joint_id, link_id) in enumerate(jacobian_pairs):
        # calculating the corresponding vector.
        active_joint_id = tree.get_active_joint_id_from_joint_id(joint_id)
        link_pose = all_link_result_cont[link_id]     # [B, 4, 4]
        joint_pose = all_joint_result_cont[active_joint_id]  # [B, 4, 4]
        
        joint_type = joint_types[joint_id]
        joint_axis_pose = all_joint_axis[:, joint_id]   # [B, 3]

        # to the world!
        joint_axis_pose_in_base = torch.bmm(joint_pose[:, :3, :3], joint_axis_pose.unsqueeze(-1)).squeeze(-1)
        offset = link_pose[:, :3, 3] - joint_pose[:, :3, 3] # [B, 3]
        
        if joint_type == 'revolute':
            out[active_joint_id, link_id * 6:link_id * 6 + 3, :] = torch.cross(joint_axis_pose_in_base, offset, dim=-1).transpose(-1, -2)
            out[active_joint_id, link_id * 6 + 3:link_id * 6 + 6, :] = joint_axis_pose_in_base.transpose(-1, -2)
        
        elif joint_type == 'prismatic':
            out[active_joint_id, link_id * 6:link+id * 6 + 3, :] = joint_axis_pose_in_base.transpose(-1, -2)

    jac_result = out.permute(2, 1, 0).contiguous()
    if ret_fk:
        return {"J": jac_result, "fk": result}
    return {"J": jac_result}


def compute_dq_new_tall(J, dx, regularization, device='cuda'):
    reg = torch.eye(J.shape[-1], device=J.device).float() * regularization
    tmpA = J.transpose(1, 2) @ J + reg 
    e = J.transpose(1, 2) @ dx.float()
    dq = torch.linalg.solve(tmpA.float(), e)
    return dq


def compute_position_ik_dx(
    link_pose, 
    target_pos,
    mask
):
    """
    link_pose:  [b, n, n_link, 3]
    targe_pose: [b, 1, n_link, 4, 4]
    mask:       [b, n_link]
    
    Returns:
    dx:         [b, n, n_link*6]
    """
    pos_err = target_pos - link_pose[:, :, :, :3, 3]        # [b, n, n_link, 3]
    rot_err = pos_err * 0.0
    err = torch.cat((pos_err, rot_err), dim=-1)             # [b, n, n_link, 6]
    mask = mask.unsqueeze(1).unsqueeze(-1).float()          # [b, 1, n_link, 1]
    err = err * mask                                        # [b, n, n_link, 6]
    return err.view(err.shape[0] * err.shape[1], -1, 1)     # [b * n, n_link * 6]


@torch.no_grad()
def batch_position_ik(
    tree, 
    target_pos, 
    mask, 
    n_retry, 
    joint_limit, 
    step_size=0.2, 
    max_iter=10, 
    regularization=1e-8
):
    batch_size = target_pos.size(0)
    device = target_pos.device
    n_dof = tree.n_dof()
    n_link = tree.n_link()

    q = torch.randn(batch_size, n_retry, n_dof).to(device) * (joint_limit[:, 1] - joint_limit[:, 0]) +  joint_limit[:, 0]

    # buffers.
    all_joint_result = torch.zeros([batch_size * n_retry, n_dof, 4, 4]).to(device)     
    all_link_result = torch.zeros([batch_size * n_retry, n_link, 4, 4]).to(device) 
    all_link_result[:, tree.get_link_id(tree.get_base_link()), :, :] = torch.eye(4).expand(batch_size * n_retry, 4, 4).to(device)
    jac_result = torch.zeros(n_dof, n_link * 6, batch_size * n_retry).to(device)

    for i in range(max_iter):
        q = q.view(-1, n_dof)
        t = time.time()
        result = batch_jacobian(
            tree,
            q, 
            fk_link_buffer=all_link_result, 
            fk_joint_buffer=all_joint_result, 
            out=jac_result,
            ret_fk=True
        )             

        J = result["J"]                                                 # [b*n, n_link*6, n_joint]
        J = J.reshape(batch_size, n_retry, -1, 6, n_dof)                # [b, n, n_link, 6, n_joint]
        J[:, :, :, 3:, :] = 0.0                                         # don't care.
        J_mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()  # [b, 1, n_link, 1, 1]
        J[:] = J * J_mask 
        J = J.view(batch_size * n_retry, n_link * 6, n_dof)

        link_pose = result["fk"]["link"]                                # [b*n, n_link, 4, 4]
        dx = compute_position_ik_dx(
            link_pose.view(batch_size, n_retry, -1, 4, 4),              # [b, n, n_link, 4, 4]
            target_pos.unsqueeze(1).view(batch_size, 1, n_link, 3),     # [b, 1, n_link, 3]
            mask                                                        # [b, n_link]
        )       
        
        t = time.time()
        dq = compute_dq_new_tall(J, dx, regularization) 
        q = q + step_size * dq.squeeze(dim=-1)
        q = torch.clip(q, joint_limit[:, 0], joint_limit[:, 1])
        # early termination
        # TODO():

    return q


def compute_contact_ik_dx(
    contact_pos0,
    contact_pos1,
    target_contact_pos0,
    target_contact_pos1,
    alpha_contact_matching=1.0
):
    """
    Args:
        contact_pos0:           [b, n, n_contact, 3]
        contact_pos1:           [b, n, n_contact, 3]
        target_contact_pos0:    [b, n, n_contact, 3]
        target_contact_pos1:    [b, n, n_contact, 3]
    
    Returns:
        dx:         [b, n, n_contact * 6]
    """
    pos_err0 = target_contact_pos0 - contact_pos0                               # [b, n, n_contact, 3]
    pos_err1 = target_contact_pos1 - contact_pos1                               # [b, n, n_contact, 3]

    err = torch.cat((pos_err0, pos_err1 * alpha_contact_matching), dim=-1)      # [b, n, n_contact, 6]
    return err.view(err.shape[0] * err.shape[1], -1)                            # [b * n, n_contact * 6]


@torch.no_grad()
def batch_contact_ik(
    tree, 
    target_pos, 
    target_normal, 
    contact_pos_in_linkf, 
    contact_normal_in_linkf, 
    contact_link_ids,   
    n_retry, 
    step_size=0.2, 
    max_iter=10, 
    regularization=1e-8,
    normal_beta=0.01,
    err_thresh=0.01,
    alpha_contact_matching=1.0,
    single_update_mode=False,
    q_init=None,
    verbose=True,
    all_joint_result_buffer=None, 
    all_link_result_buffer=None,
    jac_result_buffer=None,
    J_err_buffer=None
):
    """
    Args:
        tree:                       Kinematics tree
        target_pos:                 [batch_size, n_contact, 3]
        target_normal:              [batch_size, n_contact, 3]
        contact_pos_in_linkf:       [batch_size, n_contact, 3]
        contact_normal_in_linkf:    [batch_size, n_contact, 3]
        contact_link_ids:           [batch_size, n_contact]         The corresonding links.
        mask:                       [batch_size, n_link]
        joint_limit:                [n_dof, 2]
    
    Returns: a dict
        - q:                          [batch_size, n_joint]           # the solved result.
        - success:                    [batch_size,]                   # whether ik finds the solution successfully.
    """

    if single_update_mode:
        # in single update mode, we only update q once.
        # this mode is used for fine-grained contact adjustment.
        n_retry = 1 
        # max_iter = 1

    batch_size = target_pos.size(0)
    device = target_pos.device
    n_dof = tree.n_dof()
    n_link = tree.n_link()
    n_contact = contact_link_ids.shape[1]
    
    t = time.time()
    joint_limit_lower, joint_limit_upper = tree.get_active_joint_limit()
    # print("lower", joint_limit_lower)
    # print("upper", joint_limit_upper)

    joint_limit_lower = torch.from_numpy(joint_limit_lower).to(device)
    joint_limit_upper = torch.from_numpy(joint_limit_upper).to(device)
    
    if single_update_mode:
        q = q_init.unsqueeze(1)
    else:
        q = torch.rand(batch_size, n_retry, n_dof).to(device) * (joint_limit_upper - joint_limit_lower) + joint_limit_lower

    # buffers.
    if all_joint_result_buffer is not None:
        all_joint_result = all_joint_result_buffer * 0
    else:
        all_joint_result = torch.zeros([batch_size * n_retry, n_dof, 4, 4]).to(device)     
    
    if all_link_result_buffer is not None:
        all_link_result = all_link_result_buffer * 0
    else:
        all_link_result = torch.zeros([batch_size * n_retry, n_link, 4, 4]).to(device) 

    # inialize base link result.
    base_link_id = tree.get_link_id(tree.get_base_link())

    all_link_result[:, base_link_id, :, :].fill_(0)
    all_link_result[:, base_link_id, 0, 0] = 1
    all_link_result[:, base_link_id, 1, 1] = 1
    all_link_result[:, base_link_id, 2, 2] = 1
    all_link_result[:, base_link_id, 3, 3] = 1

    if jac_result_buffer is not None:
        jac_result = jac_result_buffer * 0
    else:
        jac_result = torch.zeros(n_dof, n_link * 6, batch_size * n_retry).to(device)

    contact_pos_in_linkf_0 = contact_pos_in_linkf                                               # this is in the corresponding link frame!
    contact_pos_in_linkf_1 = contact_pos_in_linkf + contact_normal_in_linkf * normal_beta       # this is in the corresponding link frame!

    contact_pos_in_linkf_0_exp = contact_pos_in_linkf_0.unsqueeze(1).expand(-1, n_retry, -1, -1).reshape(-1, 3, 1)    # [b*n*n_contact, 3]
    contact_pos_in_linkf_1_exp = contact_pos_in_linkf_1.unsqueeze(1).expand(-1, n_retry, -1, -1).reshape(-1, 3, 1)    # [b*n*n_contact, 3]

    target_contact_pos_0 = target_pos                                                   # [b, n_contact, 3]
    target_contact_pos_1 = target_pos + target_normal * normal_beta                             # [b, n_contact, 3]
    target_contact_pos_0_exp = target_contact_pos_0.unsqueeze(1).expand(-1, n_retry, -1, -1)    # [b, n, n_contact, 3]
    target_contact_pos_1_exp = target_contact_pos_1.unsqueeze(1).expand(-1, n_retry, -1, -1)    # [b, n, n_contact, 3]

    if J_err_buffer is not None:
        J_err = J_err_buffer * 0
    else:
        J_err = torch.zeros(batch_size, n_retry, n_contact, 6, n_dof).to(device)    # [b, n, n_contact, 6, joint]

    q = q.view(-1, n_dof)
    batch_idx = torch.arange(batch_size).to(device).unsqueeze(-1).expand(-1, n_contact)

    for i in tqdm(range(max_iter), desc="Contact IK", disable=single_update_mode or not verbose):
        t = time.time()
        result = batch_jacobian(
            tree,
            q, 
            fk_link_buffer=all_link_result, 
            fk_joint_buffer=all_joint_result, 
            out=jac_result,
            ret_fk=True
        )
        
        t = time.time()
        J = result["J"]                                                         # [b*n, n_link*6, n_joint]
        J = J.view(batch_size, n_retry, -1, 6, n_dof)                           # [b, n, n_link, 6, n_joint]
        link_pose = result["fk"]["link"]                                        # [b*n, n_link, 4, 4]
        link_pose = link_pose.view(batch_size, n_retry, n_link, 4, 4)
        link_pose_contact = link_pose[batch_idx, :, contact_link_ids, :, :]     # [b, n_contact, n, 4, 4]
        link_pose_contact = link_pose_contact.permute(0, 2, 1, 3, 4)            # [b, n, n_contact, 4, 4]

        J_contact = J[batch_idx, :, contact_link_ids, :, :]                     # [b, n_contact, n, 6, joint]
        J_contact = J_contact.permute(0, 2, 1, 3, 4)                            # [b, n, n_contact, 6, joint]

        flatten_link_rotation = link_pose_contact[..., :3, :3].reshape(-1, 3, 3)
        flatten_link_transl   = link_pose_contact[..., :3,  3].reshape(-1, 3)

        contact_pos_in_linkf_0_in_base = torch.bmm(flatten_link_rotation, contact_pos_in_linkf_0_exp).squeeze(-1)    # [b*n*n_contact, 3]
        contact_pos_in_linkf_1_in_base = torch.bmm(flatten_link_rotation, contact_pos_in_linkf_1_exp).squeeze(-1)    # [b*n*n_contact, 3]

        contact_pos_in_linkf_0_in_base_exp = contact_pos_in_linkf_0_in_base.view(batch_size, n_retry, n_contact, 3)  # [b, n, n_contact, 3]
        contact_pos_in_linkf_1_in_base_exp = contact_pos_in_linkf_1_in_base.view(batch_size, n_retry, n_contact, 3)  # [b, n, n_contact, 3]

        # v_b = v_a + w_a x r_ab.
        t = time.time()
        J_err[:, :, :, :3, :] = J_contact[:, :, :, :3, :] + torch.cross(J_contact[:, :, :, 3:, :], contact_pos_in_linkf_0_in_base_exp.unsqueeze(-1), dim=3)
        J_err[:, :, :, 3:, :] = alpha_contact_matching * (J_contact[:, :, :, :3, :] + torch.cross(J_contact[:, :, :, 3:, :], contact_pos_in_linkf_1_in_base_exp.unsqueeze(-1), dim=3))

        J_err_flatten = J_err.view(batch_size * n_retry, -1, n_dof)                     # [b*n, n_contact * 6, joint]

        contact_pos_0_in_base = contact_pos_in_linkf_0_in_base + flatten_link_transl    # [b * n * n_contact, 3]
        contact_pos_1_in_base = contact_pos_in_linkf_1_in_base + flatten_link_transl    # [b * n * n_contact, 3]
        t = time.time()
        dx = compute_contact_ik_dx(
            contact_pos_0_in_base.view(batch_size, n_retry, n_contact, 3),
            contact_pos_1_in_base.view(batch_size, n_retry, n_contact, 3),
            target_contact_pos_0_exp,
            target_contact_pos_1_exp,
            alpha_contact_matching=alpha_contact_matching
        )   # [b * n, n_contact*6]
    
        t = time.time()

        dq = compute_dq_new_tall(
            J_err_flatten.view(-1, n_contact*6, n_dof),                     # [b*n, n_contact*6, n_dof]
            dx.view(-1, n_contact*6, 1),                                    # [b*n, n_contact*6]
            regularization
        ) # [b*n, n_dof]

        q = q + step_size * dq.squeeze(dim=-1)
        q = torch.clip(q, joint_limit_lower, joint_limit_upper)
        torch.cuda.synchronize()

    
    if single_update_mode:
        chain_matrix = torch.from_numpy(tree.get_chain_matrix()).to(device)     # [n_link, n_dof]
        active_mask = chain_matrix[contact_link_ids]                            # [batch_size, n_contact, dof]

        dx = dx.view(batch_size, n_retry, n_contact, 6)
        err = (dx ** 2).sum(dim=-1) / 2                     # [b, n, n_contact]
        success_mask = (err < (err_thresh ** 2))            # [b, n, n_contact]
        success = success_mask.any(dim=1).all(dim=-1)

        err_t = err.permute(0, 2, 1)            # [b, n_contact, n]
        selection = err_t.argmin(dim=-1)        # [b, contact]
        
        # q_all[i][j] = q[batch[i][j], selection[i][j]]
        batch_idx = torch.arange(batch_size).to(device).unsqueeze(-1).expand(-1, n_contact)
        q = q.view(batch_size, n_retry, -1)
        q_all = q[batch_idx, selection] # [b, n_contact, n_dof]
        
        q = (q_all * active_mask).sum(dim=1)
        return {"q": q, "err": torch.min(err_t, axis=-1)[0], "success": None, "q_mask": None}

    # If not single update...
    # We first decompose the result based on kinematic chain, and compose the best chain-wise solutions into out final solution
    chain_matrix = torch.from_numpy(tree.get_chain_matrix()).to(device)     # [n_link, n_dof]
    active_mask = chain_matrix[contact_link_ids]                            # [batch_size, n_contact, dof]

    dx = dx.view(batch_size, n_retry, n_contact, 6)
    err = (dx ** 2).sum(dim=-1) / 2                     # [b, n, n_contact]
    success_mask = (err < (err_thresh ** 2))            # [b, n, n_contact]
    success = success_mask.any(dim=1).all(dim=-1)

    err_t = err.permute(0, 2, 1)            # [b, n_contact, n]
    selection = err_t.argmin(dim=-1)        # [b, contact]
    # q_all[i][j] = q[batch[i][j], selection[i][j]]
    batch_idx = torch.arange(batch_size).to(device).unsqueeze(-1).expand(-1, n_contact)
    q = q.view(batch_size, n_retry, -1)
    q_all = q[batch_idx, selection] # [b, n_contact, n_dof]
    q = (q_all * active_mask).sum(dim=1)
    active_mask = active_mask.sum(dim=1).bool()

    return {"q": q, "success": success, "q_mask": active_mask}
