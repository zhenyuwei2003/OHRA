import os
import sys
import viser
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

from grasp_zeroshot.utils.hand_model import HandModel


def main():
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    device = torch.device('cpu')

    urdf_dir = os.path.join(ROOT_DIR, "grasp_zeroshot/assets/urdf/")
    urdf_files = sorted([f for f in os.listdir(urdf_dir) if f.endswith(".urdf")])
    assert len(urdf_files) > 0, "No URDF files found!"

    def vis_hand(urdf_idx):
        server.scene.reset()

        urdf_file = os.path.join(urdf_dir, urdf_files[urdf_idx])
        robot_name = os.path.splitext(urdf_files[urdf_idx])[0]
        print("Loading URDF:", urdf_file)

        hand = HandModel(
            robot_name=robot_name,
            urdf_type="path",
            urdf_file=urdf_file,
            is_canonical=True,
            use_collision=False,
            device=device
        )

        server.scene.reset()

        link_trimeshes = hand.get_trimeshes_q(hand.get_initial_q())['link']

        for link_name, link_trimesh in link_trimeshes.items():
            server.scene.add_mesh_simple(
                name=f"link/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.75
            )

        for link, transform in hand.frame_status.items():
            joint_name = hand.link2joint_map.get(link, link)
            matrix = transform.get_matrix()[0].cpu().numpy()
            server.scene.add_frame(
                name=f"frame/{link}-{joint_name}",
                axes_length=0.02,
                axes_radius=0.001,
                wxyz=R.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=matrix[:3, 3]
            )

    urdf_selector = server.gui.add_slider(
        label="URDF Index",
        min=1,
        max=len(urdf_files) - 1,
        step=1,
        initial_value=1
    )
    urdf_selector.on_update(lambda _: vis_hand(urdf_selector.value))

    vis_hand(1)
    if input("Press Enter to exit...") == "":
        server.stop()


if __name__ == "__main__":
    main()
