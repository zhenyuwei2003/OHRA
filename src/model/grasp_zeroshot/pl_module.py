import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pathlib import Path
from nflows.nn.nets.resnet import ResidualNet
from scipy.spatial.transform import Rotation as R

ROOT_DIR = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT_DIR)

from utils.rotation import rot6d_to_matrix
from model.grasp_zeroshot.diffusion_model import DiffusionModel
from model.grasp_zeroshot.encoder import Encoder


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pc_encoder = Encoder(cfg.model.encoder.emb_dim)
        self.diffusion = DiffusionModel(cfg.model.diffusion)
        self.joint_network = ResidualNet(
            in_features=cfg.model.diffusion.cond_dim + cfg.model.diffusion.pose_dim,
            out_features=22,
            hidden_features=cfg.model.joint_network.hidden_features,
        )

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=False)

    def training_step(self, batch, batch_idx):
        object_pc = batch['object_pc']
        hand_params = batch['hand_params']
        q = batch['q']
        q_xyz = q[:, :3]
        q_rot = q[:, 3:9]
        q_pose = q[:, :9]
        q_joint = q[:, 9:]

        object_pc_feat = self.pc_encoder(object_pc)
        feat = torch.cat([object_pc_feat.mean(dim=1), hand_params, q_rot], dim=1)

        loss = 0.0

        pred, target = self.diffusion(x=q_xyz, cond=feat)
        assert not torch.isnan(pred).any(), f"NaN in diffusion result!"
        loss_diffusion = nn.SmoothL1Loss()(pred, target)
        loss += loss_diffusion * self.cfg.training.weight.loss_diffusion

        pred_joint = self.joint_network(torch.cat([feat, q_pose], dim=1))
        assert not torch.isnan(pred_joint).any(), f"NaN in MLP result!"
        loss_joint = nn.SmoothL1Loss()(pred_joint, q_joint)
        loss += loss_joint * self.cfg.training.weight.loss_joint

        self.log("loss", loss, prog_bar=True)
        self.log("loss_diffusion", loss_diffusion, prog_bar=False)
        self.log("loss_joint", loss_joint, prog_bar=False)

        xyz_error = F.l1_loss(pred, target, reduction='mean')
        self.log("xyz_error", xyz_error, prog_bar=False)

        joint_error = F.l1_loss(pred_joint, q_joint, reduction='mean')
        self.log("joint_error", joint_error, prog_bar=False)

        return loss

    def inference(self, object_pc, hand_params, rot6d):
        object_pc_feat = self.pc_encoder(object_pc)
        feat = torch.cat([object_pc_feat.mean(dim=1), hand_params, rot6d], dim=1)
        sampled_pose = self.diffusion.sample(cond=feat)
        pred_joint = self.joint_network(torch.cat([feat, sampled_pose, rot6d], dim=1))
        quat = torch.tensor(R.from_matrix(rot6d_to_matrix(rot6d).cpu().numpy()).as_quat(), device=rot6d.device)
        pred_q = torch.cat([sampled_pose, quat, pred_joint], dim=1)

        return pred_q.to(torch.float32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.max_epochs,
            eta_min=self.cfg.training.lr_min,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
