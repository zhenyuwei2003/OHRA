import os
import re
import sys
import json
import numpy as np
import torch
import argparse
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel


FLOAT_DIGITS = 4
def json_round(value, digits=FLOAT_DIGITS):
    round_value = round(value, digits)
    return 0.0 if round_value == -0.0 else round_value


def main(robot_name):
    meta_info = json.load(open(os.path.join(ROOT_DIR, f"assets/meta_infos/{robot_name}.json")))
    urdf_file = os.path.join(ROOT_DIR, meta_info['urdf_path']['original'])
    joint_mapping = meta_info['joint_mapping']
    palm_origin = meta_info['palm_origin']

    # get palm origin frame (x axis: palm upright, y axis: right hand thumb side, z axis: finger side)
    palm_origin_se3 = np.eye(4)
    palm_origin_se3[:3, 3] = palm_origin[:3]
    palm_origin_se3[:3, :3] = R.from_euler('xyz', palm_origin[3:]).as_matrix()

    hand = HandModel(robot_name=robot_name, urdf_type="path", urdf_file=urdf_file, is_canonical=False, device=torch.device('cpu'))
    link_trimeshes = hand.get_trimeshes_q(torch.zeros(hand.dof, dtype=torch.float32))['link']

    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # parse URDF to get joint infos
    joint_infos = {}
    for joint in root.findall('joint'):
        if joint.get('type') == 'revolute':
            origin_xyz = list(map(float, joint.find('origin').get('xyz').split())) if joint.find('origin').get('xyz') is not None else [0.0, 0.0, 0.0]
            origin_rpy = list(map(float, joint.find('origin').get('rpy').split())) if joint.find('origin').get('rpy') is not None else [0.0, 0.0, 0.0]
            joint_infos[joint.get('name')] = {
                'lower': float(joint.find('limit').get('lower')),
                'upper': float(joint.find('limit').get('upper')),
                'origin': origin_xyz + origin_rpy,
                'axis': list(map(float, joint.find('axis').get('xyz').split())),
                'parent_link': joint.find('parent').get('link'),
                'child_link': joint.find('child').get('link'),
            }

    # check joint mapping validity
    validity_flag = True
    for finger in joint_mapping:
        for joint in finger:
            if joint is not None and joint not in joint_infos:
                print(f"[ERROR] Joint <{joint}> not found in URDF!")
                validity_flag = False
    if validity_flag:
        print("[INFO] ######## URDF parsing done ########")
    else:
        exit(1)

    # param format
    json_save_infos = {
        "palm_radius": None,
        "finger_radius": None,
        "finger_lengths": [[None, None, None], [None, None, None]],
        "finger_xyz": [[None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None]],
        "thumb_rpy": [None, None, None],
        "thumb_axes": [[None, None, None], [None, None, None]],
        "little_extra_origin": [None, None, None, None, None, None],
        "joint_lowers": [[None, None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None, None]],
        "joint_uppers": [[None, None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None, None]],
    }

    # get palm link name (the link with multiple child joints)
    link_parent_count = {}
    for joint_info in joint_infos.values():
        parent_link = joint_info['parent_link']
        if parent_link not in link_parent_count:
            link_parent_count[parent_link] = 0
        link_parent_count[parent_link] += 1
    palm_list = [link for link, count in link_parent_count.items() if count > 1]
    assert len(palm_list) == 1, "Cannot determine palm link"
    palm_link_name = palm_list[0]
    # estimate palm radius using mean(length, width) of palm link bounding box
    palm_bounds = link_trimeshes[palm_link_name].bounding_box.bounds
    palm_delta = palm_bounds[1] - palm_bounds[0]
    json_save_infos["palm_radius"] = json_round((sum(palm_delta) - min(palm_delta)) / (len(palm_delta) - 1) / 2)
    print(f"[INFO] <palm_radius>: {json_save_infos['palm_radius']}")

    # get link's parent joint idx, -1 if link does not exist (corresponding joint is None)
    thumb_link_idx = [-1, -1, -1]
    thumb_link_idx[0] = 1 if joint_mapping[0][1] is not None else (0 if joint_mapping[0][0] is not None else -1)
    thumb_link_idx[1] = 3 if joint_mapping[0][3] is not None else (2 if joint_mapping[0][2] is not None else -1)
    thumb_link_idx[2] = 4 if joint_mapping[0][4] is not None else -1
    finger_link_idx = [-1, -1, -1]
    finger_link_idx[0] = 1 if joint_mapping[1][1] is not None else (0 if joint_mapping[1][0] is not None else -1)
    finger_link_idx[1] = 2 if joint_mapping[1][2] is not None else -1
    finger_link_idx[2] = 3 if joint_mapping[1][3] is not None else -1
    # estimate finger radius using min of each link bounding box, averaged over all fingers
    finger_lengths = []
    for finger_idx, finger in enumerate(joint_mapping):
        for joint_idx, joint in enumerate(finger):
            if joint is None:
                continue
            if (finger_idx == 0 and joint_idx in thumb_link_idx) or (1 <= finger_idx <= 3 and joint_idx in finger_link_idx) or (finger_idx == 4 and joint_idx - 1 in finger_link_idx):
                link_trimesh = hand.trimeshes[joint_infos[joint]['child_link']]
                link_bounds = link_trimesh.bounding_box.bounds
                link_delta = link_bounds[1] - link_bounds[0]
                finger_lengths.append(min(link_delta) / 2)
    json_save_infos["finger_radius"] = json_round(np.mean(finger_lengths))
    print(f"[INFO] <finger_radius>: {json_save_infos['finger_radius']}")

    # get frame's joint idx (next joint of first two links), -1 if frame does not exist
    thumb_frame_idx = [-1, -1]
    if joint_mapping[0][0] is None and joint_mapping[0][1] is None:  # first link does not exist
        thumb_frame_idx[0] = -1
    else:
        thumb_frame_idx[0] = 2 if joint_mapping[0][2] is not None else (3 if joint_mapping[0][3] is not None else (4 if joint_mapping[0][4] is not None else -1))
    if joint_mapping[0][2] is None and joint_mapping[0][3] is None:  # second link does not exist
        thumb_frame_idx[1] = -1
    else:
        thumb_frame_idx[1] = 4 if thumb_frame_idx[0] != 4 and joint_mapping[0][4] is not None else -1
    finger_frame_idx = [-1, -1]
    if joint_mapping[1][0] is None and joint_mapping[1][1] is None:  # first link does not exist
        finger_frame_idx[0] = -1
    else:
        finger_frame_idx[0] = 2 if joint_mapping[1][2] is not None else (3 if joint_mapping[1][3] is not None else -1)
    if joint_mapping[1][2] is None:  # second link does not exist
        finger_frame_idx[1] = -1
    else:
        finger_frame_idx[1] = 3 if finger_frame_idx[0] != 3 and joint_mapping[1][3] is not None else -1
    # estimate finger lengths using joint origin
    json_save_infos["finger_lengths"][0][0] = json_round(np.linalg.norm(joint_infos[joint_mapping[0][thumb_frame_idx[0]]]['origin'][:3])) if thumb_frame_idx[0] != -1 else 0.0  # thumb first link
    json_save_infos["finger_lengths"][0][1] = json_round(np.linalg.norm(joint_infos[joint_mapping[0][thumb_frame_idx[1]]]['origin'][:3])) if thumb_frame_idx[1] != -1 else 0.0  # thumb second link
    json_save_infos["finger_lengths"][1][0] = json_round(np.linalg.norm(joint_infos[joint_mapping[1][finger_frame_idx[0]]]['origin'][:3])) if finger_frame_idx[0] != -1 else 0.0  # finger first link
    json_save_infos["finger_lengths"][1][1] = json_round(np.linalg.norm(joint_infos[joint_mapping[1][finger_frame_idx[1]]]['origin'][:3])) if finger_frame_idx[1] != -1 else 0.0  # finger second link
    # compute average length for third link, don't use frame since tip frame is not guaranteed to exist, need manual adjustment
    json_save_infos["finger_lengths"][0][2] = json_round((json_save_infos["finger_lengths"][0][0] + json_save_infos["finger_lengths"][0][1]) / 2) if thumb_frame_idx[1] != -1 else json_save_infos["finger_lengths"][0][0]  # thumb average length
    json_save_infos["finger_lengths"][1][2] = json_round((json_save_infos["finger_lengths"][1][0] + json_save_infos["finger_lengths"][1][1]) / 2) if finger_frame_idx[1] != -1 else json_save_infos["finger_lengths"][1][0]  # finger average length
    print(f"[INFO] <finger_lengths>: {json_save_infos['finger_lengths']}")

    for finger_idx, finger in enumerate(joint_mapping):
        if all([joint is None for joint in finger]):
            json_save_infos["finger_xyz"][finger_idx] = [0.0, 0.0, 0.0]
        else:
            # get base joint idx of each finger
            base_joint_idx = 0 if finger_idx != 0 and finger_idx != 1 else 1
            max_base_joint_idx = 4 if finger_idx == 0 or finger_idx == 4 else 3
            while base_joint_idx <= max_base_joint_idx and joint_mapping[finger_idx][base_joint_idx] is None:
                base_joint_idx += 1
            base_joint = joint_mapping[finger_idx][base_joint_idx]
            base_link = joint_infos[base_joint]['child_link']
            base_link_frame = hand.frame_status[base_link].get_matrix()[0].cpu().numpy()
            finger_xyz = palm_origin_se3[:3, :3].T @ (base_link_frame[:3, 3] - palm_origin_se3[:3, 3])  # world frame to palm frame
            json_save_infos["finger_xyz"][finger_idx] = [json_round(x) for x in finger_xyz.tolist()]
    print(f"[INFO] <finger_xyz>: {json_save_infos['finger_xyz']}")

    thumb_joint_idxs = np.where(np.array(joint_mapping[0]) != None)[0]
    if len(thumb_joint_idxs) == 0:
        json_save_infos["thumb_rpy"] = [0.0, 0.0, 0.0]
    elif len(thumb_joint_idxs) == 1:
        print(f"[WARNING] Cannot determine thumb rpy, only one joint found! Please manually set <thumb_rpy> in json file.")
        json_save_infos["thumb_rpy"] = [0.0, 0.0, 0.0]
    else:
        thumb_base_joint = joint_mapping[0][thumb_joint_idxs[0]]
        thumb_tip_joint = joint_mapping[0][thumb_joint_idxs[-1]]
        thumb_base_frame = hand.frame_status[joint_infos[thumb_base_joint]['child_link']].get_matrix()[0].cpu().numpy()
        thumb_tip_frame = hand.frame_status[joint_infos[thumb_tip_joint]['child_link']].get_matrix()[0].cpu().numpy()

        # get tip-side axis (need to be +z), use thumb tip position relative to base in base frame
        delta_pos = thumb_base_frame[:3, :3].T @ (thumb_tip_frame[:3, 3] - thumb_base_frame[:3, 3])
        tip_dir = np.zeros(3)
        tip_dir[np.argmax(np.abs(delta_pos))] = np.sign(delta_pos[np.argmax(np.abs(delta_pos))])
        # get rotation axis (need to be +y), use thumb tip joint axis in base frame
        rotation_axis = np.round(thumb_tip_frame[:3, :3].T @ thumb_base_frame[:3, :3] @ np.array(joint_infos[thumb_tip_joint]['axis']))

        thumb_reorder_matrix = np.zeros((3, 3))
        thumb_reorder_matrix[:, 1] = rotation_axis
        thumb_reorder_matrix[:, 2] = tip_dir
        thumb_reorder_matrix[:, 0] = np.cross(thumb_reorder_matrix[:, 1], thumb_reorder_matrix[:, 2])
        thumb_base_frame[:3, :3] = thumb_base_frame[:3, :3] @ thumb_reorder_matrix
        thumb_base_frame_relative = np.linalg.inv(palm_origin_se3) @ thumb_base_frame

        thumb_rpy = R.from_matrix(thumb_base_frame_relative[:3, :3]).as_euler('xyz')
        json_save_infos["thumb_rpy"] = [json_round(x) for x in thumb_rpy.tolist()]
    print(f"[INFO] <thumb_rpy>: {json_save_infos['thumb_rpy']}")

    frame_status = hand.pk_chain.forward_kinematics(torch.zeros(hand.dof, dtype=torch.float32))
    # compute thumb joint1 axis
    if joint_mapping[0][0] is None:
        json_save_infos["thumb_axes"][0] = [0.0, 0.0, 0.0]
    else:
        joint1_axis = thumb_reorder_matrix.T @ np.array(joint_infos[joint_mapping[0][0]]['axis'])
        json_save_infos["thumb_axes"][0] = [json_round(x, digits=1) for x in joint1_axis.tolist()]
    # compute thumb joint2 axis
    if joint_mapping[0][1] is None:
        json_save_infos["thumb_axes"][1] = [0.0, 0.0, 0.0]
    else:
        joint2_axis = thumb_reorder_matrix.T @ np.array(joint_infos[joint_mapping[0][1]]['axis'])
        if joint_mapping[0][0] is not None:  # joint3 frame may be different from joint4 frame, compute the rotation
            matrix_joint1 = frame_status[joint_infos[joint_mapping[0][0]]['child_link']].get_matrix()[0, :3, :3].cpu().numpy()
            matrix_joint2 = frame_status[joint_infos[joint_mapping[0][1]]['child_link']].get_matrix()[0, :3, :3].cpu().numpy()
            rotation_joint2to1 = matrix_joint1.T @ matrix_joint2 
            joint2_axis = rotation_joint2to1 @ joint2_axis
        json_save_infos["thumb_axes"][1] = [json_round(x, digits=1) for x in joint2_axis.tolist()]
    print(f"[INFO] <thumb_axes>: {json_save_infos['thumb_axes']}")

    if joint_mapping[4][0] is None:
        json_save_infos["little_extra_origin"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        little_extra_joint = joint_mapping[4][0]
        little_extra_frame = hand.frame_status[joint_infos[little_extra_joint]['child_link']].get_matrix()[0].cpu().numpy()
        
        # get palm upright axis (need to be +x), (1, 0, 0) in palm frame to joint frame
        tip_dir = np.round(little_extra_frame[:3, :3].T @ np.array([1, 0, 0]))
        # get rotation axis (need to be +y)
        rotation_axis = np.round(np.array(joint_infos[little_extra_joint]['axis']))

        little_reorder_matrix = np.zeros((3, 3))
        little_reorder_matrix[:, 0] = tip_dir
        little_reorder_matrix[:, 1] = rotation_axis
        little_reorder_matrix[:, 2] = np.cross(little_reorder_matrix[:, 0], little_reorder_matrix[:, 1])
        little_extra_frame[:3, :3] = little_extra_frame[:3, :3] @ little_reorder_matrix
        little_extra_frame_relative = np.linalg.inv(palm_origin_se3) @ little_extra_frame

        little_extra_origin = np.concatenate([little_extra_frame_relative[:3, 3], R.from_matrix(little_extra_frame_relative[:3, :3]).as_euler('xyz')])
        json_save_infos["little_extra_origin"] = [json_round(x) for x in little_extra_origin.tolist()]
    print(f"[INFO] <little_extra_origin>: {json_save_infos['little_extra_origin']}")

    for finger_idx, finger in enumerate(joint_mapping):
        for joint_idx, joint in enumerate(finger):
            if joint is None:
                json_save_infos["joint_lowers"][finger_idx][joint_idx] = 0.0
                json_save_infos["joint_uppers"][finger_idx][joint_idx] = 0.0
            else:
                json_save_infos["joint_lowers"][finger_idx][joint_idx] = json_round(joint_infos[joint]['lower'])
                json_save_infos["joint_uppers"][finger_idx][joint_idx] = json_round(joint_infos[joint]['upper'])
    print(f"[INFO] <joint_lowers>: {json_save_infos['joint_lowers']}")
    print(f"[INFO] <joint_uppers>: {json_save_infos['joint_uppers']}")

    def format_json_string(json_str):  # remove extra newlines and spaces in lists
        def replacer(match):
            content = match.group(1).replace('\n', '').replace('  ', '').replace(',', ', ')
            return f"[{content}]"
        json_str = re.sub(r'\[\s*([^\[\]\{\}]*?)\s*\]', replacer, json_str)
        return json_str

    if input("Write to json file? (y/n):").strip().lower() == "y":
        print("Params saved to:", f"{ROOT_DIR}/assets/canonical/json/{robot_name}.json")
        with open(f"{ROOT_DIR}/assets/canonical/json/{robot_name}.json", 'w', encoding='utf-8') as f:
            f.write(format_json_string(json.dumps(json_save_infos, indent=4, ensure_ascii=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot")
    parser.add_argument("--extended", action="store_true", help="Whether to visualize the extended canonical hand model")
    args = parser.parse_args()

    if args.extended:
        print(f"[TODO] Parsing URDF for extended canonical hand model is not supported yet.")
        exit(1)

    main(args.robot_name)
