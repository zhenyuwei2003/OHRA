import os
import sys
import json
import viser
import torch
import trimesh
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel
from utils.rotation import rot6d_to_matrix


def create_hands(is_extended):
    json_path = os.path.join(ROOT_DIR, f"assets/meta_infos/{'extended' if is_extended else 'base'}/")
    meta_infos = {
        'allegro_hand_left': json.load(open(os.path.join(json_path, "allegro_hand_left.json"))),
        'barrett_hand': json.load(open(os.path.join(json_path, "barrett_hand.json"))),
        'shadow_hand_right': json.load(open(os.path.join(json_path, "shadow_hand_right.json")))
    }

    hands = {
        'allegro_hand_left': HandModel(
            robot_name='allegro_hand_left',
            urdf_type="path",
            urdf_file=os.path.join(ROOT_DIR, meta_infos['allegro_hand_left']['urdf_path']['canonical_extended' if is_extended else 'canonical']),
            is_canonical=True,
            is_extended=is_extended,
            use_collision=False,
            device=torch.device('cpu')
        ),
        'barrett_hand': HandModel(
            robot_name='barrett_hand',
            urdf_type="path",
            urdf_file=os.path.join(ROOT_DIR, meta_infos['barrett_hand']['urdf_path']['canonical_extended' if is_extended else 'canonical']),
            is_canonical=True,
            is_extended=is_extended,
            use_collision=False,
            device=torch.device('cpu')
        ),
        'shadow_hand_right': HandModel(
            robot_name='shadow_hand_right',
            urdf_type="path",
            urdf_file=os.path.join(ROOT_DIR, meta_infos['shadow_hand_right']['urdf_path']['canonical_extended' if is_extended else 'canonical']),
            is_canonical=True,
            is_extended=is_extended,
            use_collision=False,
            device=torch.device('cpu')
        ),
    }
    return hands


def vis_data_canonical(hands, idx):
    server.scene.reset()
    q, object_name, robot_name = metadata[idx]

    name = object_name.split('+')
    object_mesh = trimesh.load_mesh(os.path.join(ROOT_DIR, f'data/object_mesh/{name[0]}/{name[1]}/{name[1]}.stl'))
    server.scene.add_mesh_simple(
        name="object",
        vertices=object_mesh.vertices,
        faces=object_mesh.faces,
        color=(255, 111, 190),
        opacity=0.75,
    )

    canonical_link_trimeshes = hands[robot_name].get_trimeshes_q(q[9:])['link']
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
    global metadata, server

    dataset_path = os.path.join(ROOT_DIR, args.dataset_path)
    dataset = torch.load(dataset_path, map_location=torch.device('cpu'), weights_only=True)
    info, metadata = dataset['info'], dataset['metadata']

    robot_names = sorted(list(info.keys()))
    object_names = []
    for robot_name in robot_names:
        object_names.extend(list(info[robot_name]['num_per_object'].keys()))
    object_names = sorted(list(set(object_names)))

    print("******************************** <Dataset info> ********************************")
    print(f"Robot_names:\n{robot_names}\nObject_names:\n{object_names}\nTotal_num: {len(metadata)}")
    for robot_name in robot_names:
        print(f"    - {robot_name}: {info[robot_name]['num_total']}")
    print("******************************** <Dataset info> ********************************")

    hands = create_hands(args.is_extended)

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    server.scene.add_frame(
        name="world",
        axes_length=0.05,
        axes_radius=0.002,
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
    )

    vis_data_canonical(hands, 0)
    slider = server.gui.add_slider(
        label='grasp_idx',
        min=0,
        max=len(metadata) - 1,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda _: vis_data_canonical(hands, slider.value))

    if input("Press Enter to exit...") == "":
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--is_extended", action="store_true", help="Whether to use the extended canonical hand model")
    args = parser.parse_args()

    main(args)
