import os
import sys
import json
import viser
import torch
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel


def vis_hand(viser_server, hand_model, q, color=(102, 192, 255), use_trimesh_vis=False):
    link_trimeshes = hand_model.get_trimeshes_q(q)['link']

    for link_name, link_trimesh in link_trimeshes.items():
        if use_trimesh_vis:  # better mesh visualization but doesn't support opacity
            link_trimesh.visual.face_colors = [*color, 255]
            viser_server.scene.add_mesh_trimesh(
                name=f"link/{link_name}",
                mesh=link_trimesh,
            )
        else:
            viser_server.scene.add_mesh_simple(
                name=f"link/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=color,
                opacity=0.75
            )

    for link_name, transform in hand_model.frame_status.items():
        joint_name = hand_model.link2joint_map[link_name] if link_name in hand_model.link2joint_map else 'none'
        matrix = transform.get_matrix()[0].cpu().numpy()
        viser_server.scene.add_frame(
            name=f"frame/{link_name} | {joint_name}",
            axes_length=0.02,
            axes_radius=0.001,
            wxyz=R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=matrix[:3, 3]
        )


def main(args):
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    device = torch.device('cpu')

    meta_info = json.load(open(os.path.join(ROOT_DIR, f"assets/meta_infos/{args.robot_name}.json")))
    urdf_file = os.path.join(ROOT_DIR, meta_info['urdf_path']['canonical'] if args.is_canonical else meta_info['urdf_path']['original'])
    hand = HandModel(
        robot_name=args.robot_name,
        urdf_type="path",
        urdf_file=urdf_file,
        is_canonical=args.is_canonical,
        use_collision=args.use_collision,
        device=device
    )
    lower, upper = hand.get_joint_limits()
    initial_q = hand.get_initial_q()

    vis_hand(server, hand, initial_q)
    slider_list = []
    for i, joint_name in enumerate([joint.name for joint in hand.pk_chain.get_joints()]):
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower[i].item(),
            max=upper[i].item(),
            step=(upper[i] - lower[i]).item() / 100,
            initial_value=initial_q[i].item()
        )
        slider_list.append(slider)
        slider.on_update(lambda _: vis_hand(server, hand, torch.tensor([s.value for s in slider_list])))

    if input("Press Enter to exit...\n") == "":
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot")
    parser.add_argument("--is_canonical", type=bool, default=True, help="Whether to use the canonical version of the hand model")
    parser.add_argument("--use_collision", type=bool, default=False, help="Whether to visualize the collision meshes of the hand model")
    args = parser.parse_args()

    main(args)
