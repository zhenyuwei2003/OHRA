import os
import sys
import json
import viser
import torch
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from src.utils.hand_model import HandModel
from src.utils.rotation import q_euler_to_q_6d, q_6d_to_q_euler, rot6d_to_matrix


def create_hand_model(robot_name, is_canonical, is_extended, use_collision=False, device=torch.device('cpu')):
    meta_info = meta_infos[robot_name]
    urdf_file = os.path.join(ROOT_DIR, meta_info['urdf_path']['canonical_extended' if is_extended else 'canonical'] if is_canonical else meta_info['urdf_path']['original'])
    return HandModel(
        robot_name=robot_name,
        urdf_type="path",
        urdf_file=urdf_file,
        is_canonical=is_canonical,
        is_extended=is_extended,
        use_collision=use_collision,
        device=device
    )


def init(is_extended):
    global robot_name_mapping, meta_infos, hands_original, hands_canonical

    robot_name_mapping = {
        'allegro': 'allegro_hand_left',
        'barrett': 'barrett_hand',
        'shadowhand': 'shadow_hand_right'
    }

    json_path = os.path.join(ROOT_DIR, f"assets/meta_infos/{'extended' if is_extended else 'base'}/")
    meta_infos = {
        'allegro_hand_left': json.load(open(os.path.join(json_path, "allegro_hand_left.json"))),
        'barrett_hand': json.load(open(os.path.join(json_path, "barrett_hand.json"))),
        'shadow_hand_right': json.load(open(os.path.join(json_path, "shadow_hand_right.json")))
    }

    hands_original = {
        'allegro_hand_left': create_hand_model('allegro_hand_left', is_canonical=False, is_extended=False),
        'barrett_hand': create_hand_model('barrett_hand', is_canonical=False, is_extended=False),
        'shadow_hand_right': create_hand_model('shadow_hand_right', is_canonical=False, is_extended=False)
    }

    hands_canonical = {
        'allegro_hand_left': create_hand_model('allegro_hand_left', is_canonical=True, is_extended=is_extended),
        'barrett_hand': create_hand_model('barrett_hand', is_canonical=True, is_extended=is_extended),
        'shadow_hand_right': create_hand_model('shadow_hand_right', is_canonical=True, is_extended=is_extended)
    }


def data_convertion(q, robot_name):
    q_original = q.clone()
    if len(q_original) != 6 + hands_original[robot_name].dof:
        q_original = q_6d_to_q_euler(q_original)
    hand_original = hands_original[robot_name]
    hand_canonical = hands_canonical[robot_name]

    # mapping the URDF differences (URDF in dataset vs robot_urdf folder)
    if robot_name == 'barrett_hand':
        rot_matrix = R.from_euler("XYZ", q_original[3:6]).as_matrix()
        rot_correction = R.from_euler("xyz", [0, 0, 3.1416]).as_matrix()
        q_original[:3] += torch.from_numpy(rot_matrix @ np.array([0.0, 0.0, -0.0163]))
        q_original[3:6] = torch.from_numpy(R.from_matrix(rot_matrix @ rot_correction).as_euler("XYZ"))
        q_original[6:] = q_original[[8, 9, 10, 11, 12, 13, 6, 7]]
        q_original[6:] *= torch.tensor([-1, -1, -1, 1, -1, -1, -1, -1])
    elif robot_name == "shadow_hand_right":
        rot_matrix = R.from_euler("XYZ", q_original[3:6]).as_matrix()
        rot_correction = R.from_euler("xyz", [0, 0, -1.5708]).as_matrix()
        q_original[3:6] = torch.from_numpy(R.from_matrix(rot_matrix @ rot_correction).as_euler("XYZ"))

    # convert original to canonical
    q_original_se3 = np.eye(4)
    q_original_se3[:3, 3] = np.array(q_original[:3])
    q_original_se3[:3, :3] = R.from_euler("XYZ", q_original[3:6]).as_matrix()

    if robot_name == "shadow_hand_right":  # map WRJ2 and WRJ1 to wrist pose
        frame_status = hand_original.pk_chain.forward_kinematics(q_original[6:])
        palm_se3 = frame_status['ee_link'].get_matrix()[0].cpu().numpy()
        palm_se3_world = q_original_se3 @ palm_se3

        frame_status = hand_original.pk_chain.forward_kinematics(torch.zeros_like(q_original[6:]))
        palm_se3_target = frame_status['ee_link'].get_matrix()[0].cpu().numpy()
        q_original_se3 = palm_se3_world @ np.linalg.inv(palm_se3_target)

    palm_origin = meta_infos[robot_name]["palm_origin"]
    palm_matrix = np.eye(4)
    palm_matrix[:3, 3] = np.array(palm_origin[:3])
    palm_matrix[:3, :3] = R.from_euler("xyz", palm_origin[3:6]).as_matrix()

    q_canonical_se3 = q_original_se3 @ palm_matrix
    q_canonical_euler = torch.zeros(6)
    q_canonical_euler[:3] = torch.from_numpy(q_canonical_se3[:3, 3])
    q_canonical_euler[3:] = torch.from_numpy(R.from_matrix(q_canonical_se3[:3, :3]).as_euler("XYZ"))

    q_canonical = torch.cat([q_canonical_euler, torch.tensor(hand_canonical.get_canonical_ordered_q(q_original[6:]))])
    return q_original, q_canonical


def vis_data(idx):  # for align original and canonical visualization use
    server.scene.reset()

    global metadata
    q, object_name, robot_name = metadata[idx]
    if robot_name == 'ezgripper' or robot_name == 'robotiq_3finger':
        print(f"[WARNING] Visualization for {robot_name} is not required. Skipping ...")
        return
    robot_name = robot_name_mapping[robot_name]
    q_original, q_canonical = data_convertion(q, robot_name)

    hand_original = hands_original[robot_name]
    hand_canonical = hands_canonical[robot_name]

    name = object_name.split('+')
    object_mesh = trimesh.load_mesh(os.path.join(ROOT_DIR, f'data/object_mesh/{name[0]}/{name[1]}/{name[1]}.stl'))
    server.scene.add_mesh_simple(
        name="object",
        vertices=object_mesh.vertices,
        faces=object_mesh.faces,
        color=(255, 111, 190),
        opacity=0.75,
    )

    original_link_trimeshes = hand_original.get_trimeshes_q(q_original[6:])['link']
    for link_name, link_trimesh in original_link_trimeshes.items():
        server.scene.add_mesh_simple(
            name=f"original_link/{link_name}",
            vertices=link_trimesh.vertices,
            faces=link_trimesh.faces,
            position=q_original[:3].numpy(),
            wxyz=R.from_euler('XYZ', q_original[3:6].numpy()).as_quat()[[3, 0, 1, 2]],
            color=(255, 149, 71),
            opacity=0.75
        )

    canonical_link_trimeshes = hand_canonical.get_trimeshes_q(q_canonical[6:])['link']
    for link_name, link_trimesh in canonical_link_trimeshes.items():
        server.scene.add_mesh_simple(
            name=f"canonical_link/{link_name}",
            vertices=link_trimesh.vertices,
            faces=link_trimesh.faces,
            position=q_canonical[:3].numpy(),
            wxyz=R.from_euler('XYZ', q_canonical[3:6].numpy()).as_quat()[[3, 0, 1, 2]],
            color=(102, 192, 255),
            opacity=0.75
        )


def vis_data_canonical(idx):
    server.scene.reset()

    q, object_name, robot_name = metadata_canonical[idx]
    hand_canonical = hands_canonical[robot_name]

    name = object_name.split('+')
    object_mesh = trimesh.load_mesh(os.path.join(ROOT_DIR, f'data/object_mesh/{name[0]}/{name[1]}/{name[1]}.stl'))
    server.scene.add_mesh_simple(
        name="object",
        vertices=object_mesh.vertices,
        faces=object_mesh.faces,
        color=(255, 111, 190),
        opacity=0.75,
    )

    canonical_link_trimeshes = hand_canonical.get_trimeshes_q(q[9:])['link']
    for link_name, link_trimesh in canonical_link_trimeshes.items():
        server.scene.add_mesh_simple(
            name=f"canonical_link/{link_name}",
            vertices=link_trimesh.vertices,
            faces=link_trimesh.faces,
            position=q[:3].numpy(),
            wxyz=R.from_matrix(rot6d_to_matrix(q[3:9].numpy())).as_quat()[[3, 0, 1, 2]],
            color=(102, 192, 255),
            opacity=0.75
        )


def main(args):
    global metadata, metadata_canonical, server
    init(args.is_extended)

    dataset = torch.load(os.path.join(ROOT_DIR, args.dataset_path), weights_only=True)
    info, metadata = dataset['info'], dataset['metadata']
    if len(metadata[0]) == 4:
        metadata = [(m[1].cpu(), m[2], m[3]) for m in metadata]


    info_canonical = {
        'allegro_hand_left': {
            'robot_name': 'allegro_hand_left',
            'num_total': info['allegro']['num_total'],
            'num_upper_object': info['allegro']['num_upper_object'],
            'num_per_object': info['allegro']['num_per_object']
        },
        'barrett_hand': {
            'robot_name': 'barrett_hand',
            'num_total': info['barrett']['num_total'],
            'num_upper_object': info['barrett']['num_upper_object'],
            'num_per_object': info['barrett']['num_per_object']
        },
        'shadow_hand_right': {
            'robot_name': 'shadow_hand_right',
            'num_total': info['shadowhand']['num_total'],
            'num_upper_object': info['shadowhand']['num_upper_object'],
            'num_per_object': info['shadowhand']['num_per_object']
        }
    }

    metadata_canonical = []
    for q, object_name, robot_name in tqdm(metadata):
        if robot_name == 'ezgripper' or robot_name == 'robotiq_3finger':
            continue
        robot_name = robot_name_mapping[robot_name]
        q_original, q_canonical = data_convertion(q, robot_name)
        q_canonical_6d = q_euler_to_q_6d(q_canonical)
        metadata_canonical.append((q_canonical_6d, object_name, robot_name))
    dataset_canonical = {
        'info': info_canonical,
        'metadata': metadata_canonical
    }

    # visualization
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    server.scene.add_frame(
        name="world",
        axes_length=0.05,
        axes_radius=0.002,
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
    )

    vis_data_canonical(0)
    slider = server.gui.add_slider(
        label='grasp_idx',
        min=0,
        max=len(metadata_canonical) - 1,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda _: vis_data_canonical(slider.value))

    if input("Press Enter to exit...") == "":
        server.stop()

    print(f"Canonical dataset created with {len(metadata_canonical)} grasps, saving to: {ROOT_DIR}/data/CMapDataset_filtered/cmap_dataset_{'extended_' if args.is_extended else ''}canonical.pt")
    if input("Do you want to save the canonical dataset? (y/n) ") == "y":
        torch.save(dataset_canonical, f"{ROOT_DIR}/data/CMapDataset_filtered/cmap_dataset_{'extended_' if args.is_extended else ''}canonical.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--is_extended", action="store_true", help="Whether to use the extended canonical hand model")
    args = parser.parse_args()

    main(args)
