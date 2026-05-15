import os
import sys
import time
import viser
import torch
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

from grasp_zeroshot.utils.hand_model import HandModel
from utils.rotation import rot6d_to_matrix


def main():
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    device = torch.device('cpu')

    data = torch.load(os.path.join(ROOT_DIR, f"data/grasp_zeroshot/leap_dataset_geq8_max200.pt"), weights_only=True)
    robot_name_list = sorted(list(set([item[2] for item in data])))
    data_subsets = {}
    for robot_name in robot_name_list:
        data_subsets[robot_name] = [item for item in data if item[2] == robot_name]

    def visualize(hand_idx, grasp_idx):
        server.scene.reset()

        robot_name = robot_name_list[hand_idx]
        data = data_subsets[robot_name]
        idx = grasp_idx % len(data)
        q, object_name, _ = data[idx]

        hand = HandModel(
            robot_name=robot_name,
            urdf_type="path",
            urdf_file=os.path.join(ROOT_DIR, f"grasp_zeroshot/assets/urdf/{robot_name}.urdf"),
            is_canonical=True,
            use_collision=False,
            device=device
        )

        object_trimesh = trimesh.load_mesh(os.path.join(ROOT_DIR, f"data/grasp_zeroshot/object/{object_name}.stl"))
        server.scene.add_mesh_simple(
            name="object",
            vertices=object_trimesh.vertices,
            faces=object_trimesh.faces,
            color=(255, 149, 71),
            opacity=0.6,
        )

        link_trimeshes = hand.get_trimeshes_q(q[9:])['link']
        for link_name, link_trimesh in link_trimeshes.items():
            server.scene.add_mesh_simple(
                name=f"canonical_link/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.75,
                position=q[:3].numpy(),
                wxyz=R.from_matrix(rot6d_to_matrix(q[3:9]).numpy()).as_quat()[[3, 0, 1, 2]]
            )

    visualize(0, 0)

    robot_slider = server.gui.add_slider(
        label="hand_idx",
        min=0,
        max=len(robot_name_list) - 1,
        step=1,
        initial_value=0,
    )
    grasp_slider = server.gui.add_slider(
        label="grasp_idx",
        min=0,
        max=199,
        step=1,
        initial_value=0,
    )
    robot_slider.on_update(lambda _: visualize(robot_slider.value, grasp_slider.value))
    grasp_slider.on_update(lambda _: visualize(robot_slider.value, grasp_slider.value))

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
