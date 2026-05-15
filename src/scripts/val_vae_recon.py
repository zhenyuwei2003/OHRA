import os
import sys
import json
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from model.vae import TrainingModule
from utils.urdf_render import urdf_render
from data_utils.sample_utils import *


def main(args):
    cfg = OmegaConf.load(f"{ROOT_DIR}/output/{args.ckpt_name}/log/hydra/.hydra/config.yaml")
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    module = TrainingModule.load_from_checkpoint(
        checkpoint_path=f"{ROOT_DIR}/output/{args.ckpt_name}/state_dict/epoch={args.epoch}.ckpt",
        map_location=args.device,
        cfg=cfg.model,
    )
    module.eval()

    params_dict = json.load(open(f"{ROOT_DIR}/assets/canonical/json/{args.robot_name}.json"))
    print("-------------------------------- [Input Params] --------------------------------")
    for k, v in params_dict.items():
        print(f"{k}: {v}")
    print("-------------------------------- [Input Params] --------------------------------")
    params_list = params_dict_to_list(params_dict)
    params_data = torch.tensor(params_list_to_data(params_list), dtype=torch.float32, device=args.device)

    recon_params, _, _ = module.model(params_data.unsqueeze(0))

    recon_params_list = params_data_to_list(recon_params.squeeze(0).detach().cpu().numpy().tolist(), joint_ranges=params_list[-44:])
    recon_params_dict = params_list_to_dict(recon_params_list)
    print("-------------------------------- [Reconstructed Params] --------------------------------")
    for k, v in recon_params_dict.items():
        print(f"{k}: {v}")
    print("-------------------------------- [Reconstructed Params] --------------------------------")

    os.makedirs(f"{ROOT_DIR}/vae_urdf", exist_ok=True)
    urdf_render(recon_params_dict, f"{ROOT_DIR}/vae_urdf/recon_{args.robot_name}.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, required=True, help="Name of the checkpoint to load")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch of the checkpoint to load")
    parser.add_argument("--robot_name", type=str, required=True, help="Name of the robot to reconstruct")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    args = parser.parse_args()

    main(args)
