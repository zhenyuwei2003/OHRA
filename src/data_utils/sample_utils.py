import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


params_dict_template = {
    "palm_radius": None,
    "finger_radius": None,
    "finger_lengths": [[None, None, None], [None, None, None]],
    "finger_xyz": [[None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None]],
    "little_extra_origin": [None, None, None, None, None, None],
    "thumb_rpy": [None, None, None],
    "thumb_axes": [[None, None, None], [None, None, None]],
    "joint_lowers": [[None, None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None, None]],
    "joint_uppers": [[None, None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None, None]],
}


def params_dict_to_list(params_dict):
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item
    
    params_list = []
    for value in params_dict.values():
        if isinstance(value, float):
            params_list.append(value)
        else:
            params_list.extend(list(flatten(value)))
    return params_list


def params_list_to_dict(params_list):
    def fill_dict(param, values_iter):
        if isinstance(param, dict):
            return {k: fill_dict(v, values_iter) for k, v in param.items()}
        elif isinstance(param, list):
            return [fill_dict(v, values_iter) for v in param]
        elif param is None:
            return next(values_iter)
        else:
            return param

    params_dict = fill_dict(params_dict_template, iter(params_list))
    return params_dict


def sample_params_dict():
    params_dict = params_dict_template.copy()

    is_right_hand = np.random.choice([False, True])
    # print("Right hand:" if is_right_hand else "Left hand:")
    use_fingers = np.random.choice([False, True], size=5, p=[0.25, 0.75])
    while sum(use_fingers[1:]) < 2:
        use_fingers[np.random.choice([1, 2, 3, 4])] = True
    use_fingers[0] = True  # Always use thumb
    finger_num = sum(use_fingers)

    use_joints = np.zeros(22, dtype=int)
    thumb_joints_choice = [[1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]]
    use_joints[:5] = thumb_joints_choice[np.random.choice(len(thumb_joints_choice))]
    use_finger_joints = np.zeros(4, dtype=int)
    while sum(use_finger_joints[1:]) < 2:
        use_finger_joints = np.random.choice([0, 1], size=4)
    if use_finger_joints[0]:
        use_joints[5] = use_joints[9] = use_joints[13] = use_joints[18] = 1
    if use_finger_joints[1]:
        use_joints[6] = use_joints[10] = use_joints[14] = use_joints[19] = 1
    if use_finger_joints[2]:
        use_joints[7] = use_joints[11] = use_joints[15] = use_joints[20] = 1
    if use_finger_joints[3]:
        use_joints[8] = use_joints[12] = use_joints[16] = use_joints[21] = 1
    if use_fingers[1] == False:
        use_joints[5:9] = 0
    if use_fingers[2] == False:
        use_joints[9:13] = 0
    if use_fingers[3] == False:
        use_joints[13:17] = 0
    if use_fingers[4] == False:
        use_joints[17:22] = 0
    if sum(use_joints) >= 20:
        use_joints[17] = np.random.choice([0, 1], p=[0.2, 0.8])
    # print("Thumb:", use_joints[:5], "Index:", use_joints[5:9], "Middle:", use_joints[9:13], "Ring:", use_joints[13:17], "Little:", use_joints[17:22])

    params_dict["palm_radius"] = np.random.uniform(0.04, 0.06)
    params_dict["finger_radius"] = np.random.uniform(0.0075, min(0.012, params_dict["palm_radius"] / 4))

    params_dict["finger_lengths"][0] = np.random.uniform(0.035, 0.07, size=3).tolist()
    params_dict["finger_lengths"][1] = np.random.uniform(0.025, 0.07, size=3).tolist()
    
    params_dict["finger_xyz"][0][0] = np.random.uniform(-0.02, 0.02)
    finger_x = np.random.normal(0.0, 0.001)
    params_dict["finger_xyz"][1][0] = finger_x
    params_dict["finger_xyz"][2][0] = finger_x
    params_dict["finger_xyz"][3][0] = finger_x
    params_dict["finger_xyz"][4][0] = finger_x
    params_dict["finger_xyz"][0][1] = np.random.uniform(-0.04, 0.0)

    k = finger_num - 1
    d = 2 * params_dict["finger_radius"]
    total_space = 2 * (params_dict["palm_radius"] - params_dict["finger_radius"]) - (k - 1) * d
    assert total_space > 0, f"{params_dict['palm_radius']}, {params_dict['finger_radius']}, {k}, {d}"
    gaps = np.random.dirichlet(np.ones(k + 1)) * total_space
    finger_ys = params_dict["finger_radius"] - params_dict["palm_radius"] + np.cumsum(gaps[:-1]) + np.arange(k) * d
    params_dict["finger_xyz"][1][1] = np.random.uniform(params_dict["finger_radius"] - params_dict["palm_radius"], params_dict["palm_radius"] - params_dict["finger_radius"])
    params_dict["finger_xyz"][2][1] = np.random.uniform(params_dict["finger_radius"] - params_dict["palm_radius"], params_dict["palm_radius"] - params_dict["finger_radius"])
    params_dict["finger_xyz"][3][1] = np.random.uniform(params_dict["finger_radius"] - params_dict["palm_radius"], params_dict["palm_radius"] - params_dict["finger_radius"])
    params_dict["finger_xyz"][4][1] = np.random.uniform(params_dict["finger_radius"] - params_dict["palm_radius"], params_dict["palm_radius"] - params_dict["finger_radius"])
    y_idx = 0
    for idx, flag in enumerate(use_fingers[1:]):
        if flag:
            params_dict["finger_xyz"][idx + 1][1] = finger_ys[y_idx]
            y_idx += 1
    if is_right_hand:
        for i in range(5):
            params_dict["finger_xyz"][i][1] = -params_dict["finger_xyz"][i][1]
    params_dict["finger_xyz"][0][2] = np.random.uniform(0.015, 0.025)
    for i in range(1, 5):
        params_dict["finger_xyz"][i][2] = np.random.normal(params_dict["palm_radius"] * 2 - 0.01, 0.005)

    params_dict["thumb_rpy"] = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]

    axis_options = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    params_dict["thumb_axes"][0] = list(np.array(axis_options[np.random.choice([0, 1, 2])]) * np.random.choice([-1, 1]))
    params_dict["thumb_axes"][1] = list(np.array(axis_options[np.random.choice([0, 1, 2])]) * np.random.choice([-1, 1]))
    while params_dict["thumb_axes"][1] == params_dict["thumb_axes"][0]:
        params_dict["thumb_axes"][1] = list(np.array(axis_options[np.random.choice([0, 1, 2])]) * np.random.choice([-1, 1]))

    params_dict["little_extra_origin"][0] = np.random.uniform(-0.01, 0.01)
    params_dict["little_extra_origin"][1] = np.random.uniform(0.03, 0.04) if is_right_hand else np.random.uniform(-0.04, -0.03)
    params_dict["little_extra_origin"][2] = np.random.uniform(0.018, 0.025)
    params_dict["little_extra_origin"][3] = np.random.uniform(5 * np.pi / 16, 7 * np.pi / 16) if is_right_hand else np.random.uniform(-7 * np.pi / 16, -5 * np.pi / 16)
    params_dict["little_extra_origin"][4] = 0.0
    params_dict["little_extra_origin"][5] = 0.0

    idx = 0
    for finger_idx in range(5):
        for joint_idx in range(len(params_dict["joint_lowers"][finger_idx])):
            if use_joints[idx]:
                params_dict["joint_lowers"][finger_idx][joint_idx] = 0.0
                params_dict["joint_uppers"][finger_idx][joint_idx] = np.pi / 2
            else:
                params_dict["joint_lowers"][finger_idx][joint_idx] = 0.0
                params_dict["joint_uppers"][finger_idx][joint_idx] = 0.0
            idx += 1
    return params_dict


def sample_params_list():
    params_list = params_dict_to_list(sample_params_dict())
    return params_list


def params_list_to_data(params_list):
    axis_mapping = {
        (0, 0, 0): [1, 0, 0, 0, 0, 0],
        (1, 0, 0): [1, 0, 0, 0, 0, 0],
        (-1, 0, 0): [0, 1, 0, 0, 0, 0],
        (0, 1, 0): [0, 0, 1, 0, 0, 0],
        (0, -1, 0): [0, 0, 0, 1, 0, 0],
        (0, 0, 1): [0, 0, 0, 0, 1, 0],
        (0, 0, -1): [0, 0, 0, 0, 0, 1],
    }
    params_cont, params_axis, params_range = params_list[:32], params_list[32:38], params_list[38:]
    data_axis = axis_mapping[tuple(params_axis[:3])] + axis_mapping[tuple(params_axis[3:])]
    data_disc = [1.0 if params_range[i] != params_range[i + 22] else 0.0 for i in range(22)]
    data = params_cont + data_axis + data_disc
    return data


def params_data_to_list(data: list, joint_ranges: list = None):
    params_cont, data_axis, data_disc = data[:32], data[32:44], data[44:]

    axis_options = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    axis1_prob = torch.softmax(torch.tensor(data_axis[:6]), dim=0).numpy().tolist()
    axis2_prob = torch.softmax(torch.tensor(data_axis[6:]), dim=0).numpy().tolist()
    axis1 = axis_options[axis1_prob.index(max(axis1_prob))]
    axis2 = axis_options[axis2_prob.index(max(axis2_prob))]
    params_axis = axis1 + axis2

    params_range = []
    if joint_ranges is not None:
        params_range.extend(joint_ranges[i] if data_disc[i] > 0.5 else 0.0 for i in range(22))
        params_range.extend(joint_ranges[22 + i] if data_disc[i] > 0.5 else 0.0 for i in range(22))
    else:
        params_range.extend([0.0] * 22)
        params_range.extend([np.pi / 2 if data_disc[i] > 0.5 else 0.0 for i in range(22)])

    params_list = params_cont + params_axis + params_range
    return params_list


def sample_data():
    params_list = sample_params_list()
    data = params_list_to_data(params_list)
    return data
