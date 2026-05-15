import os
import sys
import json
import viser
import torch
import trimesh
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from utils.hand_model import HandModel
from utils.rotation import matrix_to_rot6d
from model.grasp.pl_module import TrainingModule
from data_utils.sample_utils import params_dict_to_list, params_list_to_data


def main(args):
    cfg = OmegaConf.load(f"{ROOT_DIR}/output/{args.ckpt_name}/log/hydra/.hydra/config.yaml")

    pl_module = TrainingModule.load_from_checkpoint(
        checkpoint_path=f"{ROOT_DIR}/output/{args.ckpt_name}/state_dict/epoch={args.epoch}.ckpt",
        map_location=args.device,
        cfg=cfg,
    )
    pl_module.eval()

    params_dict = json.load(open(os.path.join(ROOT_DIR, f"assets/canonical/json/{args.robot_name}.json")))
    hand_params = torch.tensor(params_list_to_data(params_dict_to_list(params_dict)), dtype=torch.float32, device=args.device).unsqueeze(0)  # (1, 66)

    object_name_split = args.object_name.split('+')
    object_trimesh = trimesh.load_mesh(os.path.join(ROOT_DIR, f'data/object_mesh/{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.stl'))
    object_pc = torch.load(os.path.join(ROOT_DIR, f'data/object_pc/{object_name_split[0]}/{object_name_split[1]}.pt'), weights_only=True)[:, :3]
    object_pc = object_pc.to(dtype=torch.float32, device=args.device).unsqueeze(0)  # (1, 512, 3)

    hand = HandModel(
        robot_name=args.robot_name,
        urdf_type="path",
        urdf_file=f"{ROOT_DIR}/assets/canonical/urdf/{args.robot_name}.urdf",
        is_canonical=True,
        use_collision=False,
        device=args.device
    )
    lower, upper = hand.get_joint_limits()

    server = viser.ViserServer(host='127.0.0.1', port=8080)
    server.scene.add_mesh_simple(
        name="object",
        vertices=object_trimesh.vertices,
        faces=object_trimesh.faces,
        color=(255, 111, 190),
        opacity=0.75,
    )

    def on_click():
        with torch.no_grad():
            rot_matrix = torch.tensor(R.random(1).as_matrix(), dtype=torch.float32)  # (1, 3, 3)
            rot_6d = matrix_to_rot6d(rot_matrix).to(args.device)
            predict_q = pl_module.inference(object_pc, hand_params, rot_6d)[0]
            predict_q[7:] = torch.clamp(predict_q[7:], lower, upper)

        print("=" * 80)
        print("Robot:", args.robot_name)
        print("Object:", args.object_name)
        print("Predict q: \n", predict_q)
        print("=" * 80)

        link_trimeshes = hand.get_trimeshes_q(predict_q[7:])['link']
        for link_name, link_trimesh in link_trimeshes.items():
            server.scene.add_mesh_simple(
                name=f"predict/{link_name}",
                vertices=link_trimesh.vertices,
                faces=link_trimesh.faces,
                color=(102, 192, 255),
                opacity=0.5,
                position=predict_q[:3].cpu().numpy(),
                wxyz=predict_q[3:7][[3, 0, 1, 2]].cpu().numpy()
            )

    button = server.gui.add_button(label='Inference')
    button.on_click(lambda _: on_click())

    if input("Press Enter to exit...\n") == "":
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, required=True, help="Name of the checkpoint to load")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch of the checkpoint to load")
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot to visualize")
    parser.add_argument("--object_name", type=str, required=True, help="Name of the object to visualize")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    args = parser.parse_args()

    main(args)
