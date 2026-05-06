import sys
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from src.utils.rotation import rot6d_to_matrix


def controller(hand, q, outer_coeff=0.1, inner_coeff=0.1):
    q_origin = torch.atleast_2d(q)[:, :-22]
    q_joint = torch.atleast_2d(q)[:, -22:]

    frame_status = hand.pk_chain.forward_kinematics(q_joint)
    lower_q, upper_q = hand.pk_chain.get_joint_limits()

    outer_q_batch = []
    inner_q_batch = []
    for batch_idx in range(q_joint.shape[0]):
        origin_se3 = torch.eye(4)
        origin_se3[:3, 3] = q_origin[batch_idx, :3]
        if q_origin.shape[1] == 9:
            origin_se3[:3, :3] = rot6d_to_matrix(q_origin[batch_idx, 3:])
        elif q_origin.shape[1] == 6:
            r = R.from_euler('XYZ', q_origin[batch_idx, 3:].cpu().numpy())
            origin_se3[:3, :3] = torch.tensor(r.as_matrix(), dtype=torch.float32)

        joint_dots = {}
        for frame_name in hand.pk_chain.get_frame_names():
            frame = hand.pk_chain.find_frame(frame_name)

            frame_se3 = origin_se3 @ frame_status[frame_name].get_matrix()[batch_idx]
            axis_dir = frame_se3[:3, :3] @ frame.joint.axis
            link_dir = frame_se3[:3, :3] @ torch.tensor([0, 0, 1], dtype=torch.float32)  # canonical URDF format assumption
            normal_dir = torch.cross(axis_dir, link_dir, dim=0)
            axis_origin = frame_se3[:3, 3]
            origin_dir = -axis_origin / torch.norm(axis_origin)
            joint_dots[frame.joint.name] = torch.dot(normal_dir, origin_dir)

        outer_q, inner_q = q_joint[batch_idx].clone(), q_joint[batch_idx].clone()
        for joint_name, dot in joint_dots.items():
            idx = hand.joint_orders.index(joint_name)
            outer_q[idx] += outer_coeff * ((lower_q[idx] - outer_q[idx]) if dot >= 0 else (upper_q[idx] - outer_q[idx]))
            inner_q[idx] += inner_coeff * ((upper_q[idx] - inner_q[idx]) if dot >= 0 else (lower_q[idx] - inner_q[idx]))
        outer_q_batch.append(outer_q)
        inner_q_batch.append(inner_q)

    outer_q_batch = torch.stack(outer_q_batch, dim=0)
    inner_q_batch = torch.stack(inner_q_batch, dim=0)

    if q.ndim == 2:  # batch
        return outer_q_batch.to(q.device), inner_q_batch.to(q.device)
    else:
        return outer_q_batch[0].to(q.device), inner_q_batch[0].to(q.device)
