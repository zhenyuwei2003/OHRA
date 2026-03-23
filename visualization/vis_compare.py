import os
import sys
import json
import viser
import torch
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel


def main(args):
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    device = torch.device('cpu')

    meta_info = json.load(open(os.path.join(ROOT_DIR, f"assets/meta_infos/{args.robot_name}.json")))
    original_urdf_file = os.path.join(ROOT_DIR, meta_info['urdf_path']['original'])
    if args.extended:
        canonical_urdf_file = os.path.join(ROOT_DIR, meta_info['urdf_path']['canonical_extended'])
    else:
        canonical_urdf_file = os.path.join(ROOT_DIR, meta_info['urdf_path']['canonical'])
    hand_original = HandModel(
        robot_name=args.robot_name,
        urdf_type="path",
        urdf_file=original_urdf_file,
        is_canonical=False,
        use_collision=args.use_collision,
        device=device
    )
    hand_canonical = HandModel(
        robot_name=args.robot_name,
        urdf_type="path",
        urdf_file=canonical_urdf_file,
        is_canonical=True,
        use_collision=args.use_collision,
        device=device
    )
    lower, upper = hand_original.get_joint_limits()
    initial_q = hand_original.get_initial_q()

    palm_origin = meta_info["palm_origin"]
    palm_origin_se3 = np.eye(4)
    palm_origin_se3[:3, 3] = palm_origin[:3]
    palm_origin_se3[:3, :3] = R.from_euler('xyz', palm_origin[3:]).as_matrix()
    palm_origin_se3_inv = np.linalg.inv(palm_origin_se3)

    def visualize(q):
        # original hand visualization
        original_link_trimeshes = hand_original.get_trimeshes_q(q)['link']
        for link_trimesh in original_link_trimeshes.values():
            link_trimesh.apply_transform(palm_origin_se3_inv)  # compensate hand palm origin difference between original and canonical

        for link_name, link_trimesh in original_link_trimeshes.items():
            server.scene.add_mesh_simple(
                name=f"link_original/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=(255, 149, 71),
                opacity=0.75,
            )
        for link_name, transform in hand_original.frame_status.items():
            joint_name = hand_original.link2joint_map[link_name] if link_name in hand_original.link2joint_map else 'none'
            matrix = transform.get_matrix()[0].cpu().numpy()
            matrix = palm_origin_se3_inv @ matrix
            server.scene.add_frame(
                name=f"frame_original/{link_name} | {joint_name}",
                axes_length=0.02,
                axes_radius=0.001,
                wxyz=R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=matrix[:3, 3],
                origin_color=(255, 149, 71)
            )

        # canonical hand visualization
        canonical_q = hand_canonical.get_canonical_ordered_q(q)
        canonical_link_trimeshes = hand_canonical.get_trimeshes_q(canonical_q)['link']
        for link_name, link_trimesh in canonical_link_trimeshes.items():
            server.scene.add_mesh_simple(
                name=f"link_canonical{'_extended' if args.extended else ''}/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.75
            )
        for link_name, transform in hand_canonical.frame_status.items():
            joint_name = hand_canonical.link2joint_map[link_name] if link_name in hand_canonical.link2joint_map else 'none'
            matrix = transform.get_matrix()[0].cpu().numpy()
            server.scene.add_frame(
                name=f"frame_canonical{'_extended' if args.extended else ''}/{link_name} | {joint_name}",
                axes_length=0.02,
                axes_radius=0.001,
                wxyz=R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=matrix[:3, 3],
                origin_color=(102, 192, 255)
            )

    visualize(initial_q)
    slider_list = []
    for i, joint_name in enumerate([joint.name for joint in hand_original.pk_chain.get_joints()]):
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower[i].item(),
            max=upper[i].item(),
            step=(upper[i] - lower[i]).item() / 100,
            initial_value=initial_q[i].item()
        )
        slider_list.append(slider)
        slider.on_update(lambda _: visualize(torch.tensor([s.value for s in slider_list])))

    if input("Press Enter to exit...\n") == "":
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot")
    parser.add_argument("--use_collision", action="store_true", help="Whether to visualize the collision meshes of the hand model")
    parser.add_argument("--extended", action="store_true", help="Whether to visualize the extended canonical hand model")
    args = parser.parse_args()

    main(args)
