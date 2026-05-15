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


def recon_hand(module, input_path, output_path):
    params_dict = json.load(open(input_path))
    params_list = params_dict_to_list(params_dict)
    params_data = torch.tensor(params_list_to_data(params_list), dtype=torch.float32, device=args.device)

    z = module.model.reparameterize(*module.model.encode(params_data.unsqueeze(0))).squeeze(0)
    recon_params_data = module.model.decode(z.unsqueeze(0)).squeeze(0).detach().cpu().numpy().tolist()
    recon_params_list = params_data_to_list(recon_params_data, joint_ranges=params_list[-44:])
    recon_params_dict = params_list_to_dict(recon_params_list)

    urdf_render(recon_params_dict, output_path)
    return z


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

    os.makedirs(f"{ROOT_DIR}/vae_urdf", exist_ok=True)

    # z_leap = recon_hand(
    #     module,
    #     f"{ROOT_DIR}/assets/canonical/json/leap_hand_right.json",
    #     f"{ROOT_DIR}/latent_leap_hand_right.urdf"
    # )
    #
    # z_allegro = recon_hand(
    #     module,
    #     f"{ROOT_DIR}/assets/canonical/json/allegro_hand_right.json",
    #     f"{ROOT_DIR}/latent_allegro_hand_right.urdf"
    # )

    z_dex3 = recon_hand(
        module,
        f"{ROOT_DIR}/assets/canonical/json/dex3_right.json",
        f"{ROOT_DIR}/vae_urdf/latent_dex3_right.urdf"
    )

    z_shadow = recon_hand(
        module,
        f"{ROOT_DIR}/assets/canonical/json/shadow_hand_right.json",
        f"{ROOT_DIR}/vae_urdf/latent_shadow_hand_right.urdf"
    )

    n_interp = 8
    z = torch.zeros((n_interp, cfg.model.latent_dim), device=args.device)
    for i in range(n_interp):
        alpha = i / (n_interp - 1)
        z[i] = (1 - alpha) * z_dex3 + alpha * z_shadow

    recon_params = module.model.decode(z).detach().cpu().numpy().tolist()

    for i in range(n_interp):
        params = recon_params[i]
        params_list = params_data_to_list(params)
        params_dict = params_list_to_dict(params_list)
        urdf_render(params_dict, f"{ROOT_DIR}/vae_urdf/latent_interp_sample_{i}.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, required=True, help="Name of the checkpoint to load")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch of the checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    args = parser.parse_args()

    main(args)
