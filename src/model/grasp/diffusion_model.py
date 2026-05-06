import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from diffusers import DDPMScheduler, DDIMScheduler

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

from model.grasp.diffusion_backbone import MLPWrapper


class DiffusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q_dim = cfg.q_dim
        self.num_train_timesteps = cfg.scheduler.num_train_timesteps
        self.prediction_type = cfg.scheduler.prediction_type

        self.backbone = MLPWrapper(
            channels=3,
            feature_dim=cfg.cond_dim,
            hidden_layers_dim=cfg.hidden_layers_dim,
            output_dim=3
        )

        self.scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=cfg.scheduler.num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=cfg.scheduler.clip_sample,
            prediction_type=cfg.scheduler.prediction_type
        )
        self.scheduler_ddim = DDIMScheduler(
            num_train_timesteps=cfg.scheduler.num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=cfg.scheduler.clip_sample,
            prediction_type=cfg.scheduler.prediction_type
        )

    def forward(self, x, cond):
        device = x.device
        B = x.shape[0]

        timesteps = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=device)
        noise = torch.randn_like(x, device=device)

        noisy_x = self.scheduler_ddpm.add_noise(x, noise, timesteps)
        pred = self.backbone(noisy_x, timesteps / self.num_train_timesteps, cond=cond)

        if self.prediction_type == 'v_prediction':
            target = self.scheduler_ddpm.get_velocity(x, noise, timesteps)
        elif self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'sample':
            target = x
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")

        return pred, target

    def sample(self, cond):
        self.scheduler_ddpm.set_timesteps(num_inference_steps=200)
        device = cond.device
        B = cond.shape[0]

        with torch.no_grad():
            noisy_x = torch.randn((B, 3), device=device)

            for t in self.scheduler_ddpm.timesteps:  # inverse order
                t_in = torch.full((B,), t / self.num_train_timesteps, device=device, dtype=torch.float32)
                pred = self.backbone(noisy_x, t_in, cond=cond)

                noisy_x = self.scheduler_ddpm.step(
                    model_output=pred,
                    timestep=t,
                    sample=noisy_x,
                ).prev_sample

        return noisy_x
