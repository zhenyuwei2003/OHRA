import sys
import viser
import torch
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel as HandModel_original
from grasp_zeroshot.utils.hand_model import HandModel


def main(args):
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    device = torch.device('cpu')

    hand_original = HandModel_original(
        robot_name="leap_hand_right",
        urdf_type="path",
        urdf_file=f"{ROOT_DIR}/assets/robot_urdf/leap_hand/leap_hand_right.urdf",
        is_canonical=False,
        is_extended=False,
        use_collision=False,
        device=device
    )
    hand_canonical = HandModel(
        robot_name=args.robot_name,
        urdf_type="path",
        urdf_file=f"{ROOT_DIR}/grasp_zeroshot/assets/urdf/{args.robot_name}.urdf",
        is_canonical=True,
        is_extended=False,
        use_collision=False,
        device=device
    )
    original_lower, original_upper = hand_original.get_joint_limits()
    canonical_lower, canonical_upper = hand_canonical.get_joint_limits()
    original_initial_q = hand_original.get_initial_q()

    palm_origin = hand_original.meta_info["palm_origin"]
    palm_origin_se3 = np.eye(4)
    palm_origin_se3[:3, 3] = palm_origin[:3]
    palm_origin_se3[:3, :3] = R.from_euler('xyz', palm_origin[3:]).as_matrix()
    palm_origin_se3_inv = np.linalg.inv(palm_origin_se3)

    def visualize(q, print_frame_info=False):
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
            if print_frame_info:
                print(f"[Original: {link_name} | {joint_name}] xyz: {matrix[:3, 3]}, rpy: {R.from_matrix(matrix[:3, :3]).as_euler('xyz')}")

        if print_frame_info:
            print("=" * 64)

        # canonical hand visualization
        canonical_q = hand_canonical.get_canonical_ordered_q(q, canonical_order=[13, -14, 0, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 9, 10, 11, 12])
        canonical_link_trimeshes = hand_canonical.get_trimeshes_q(canonical_q)['link']
        for link_name, link_trimesh in canonical_link_trimeshes.items():
            server.scene.add_mesh_simple(
                name=f"link_canonical/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.75
            )
        for link_name, transform in hand_canonical.frame_status.items():
            joint_name = hand_canonical.link2joint_map[link_name] if link_name in hand_canonical.link2joint_map else 'none'
            matrix = transform.get_matrix()[0].cpu().numpy()
            server.scene.add_frame(
                name=f"frame_canonical/{link_name} | {joint_name}",
                axes_length=0.02,
                axes_radius=0.001,
                wxyz=R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=matrix[:3, 3],
                origin_color=(102, 192, 255)
            )
            if print_frame_info:
                print(f"[Canonical: {link_name} | {joint_name}] xyz: {matrix[:3, 3]}, rpy: {R.from_matrix(matrix[:3, :3]).as_euler('xyz')}")

    visualize(original_initial_q, print_frame_info=True)
    slider_list = []
    for i, joint_name in enumerate([joint.name for joint in hand_original.pk_chain.get_joints()]):
        slider = server.gui.add_slider(
            label=joint_name,
            min=original_lower[i].item(),
            max=original_upper[i].item(),
            step=(original_upper[i] - original_lower[i]).item() / 100,
            initial_value=original_initial_q[i].item()
        )
        slider_list.append(slider)
        slider.on_update(lambda _: visualize(torch.tensor([s.value for s in slider_list])))

    if input("Press Enter to exit...\n") == "":
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot")
    args = parser.parse_args()

    main(args)
