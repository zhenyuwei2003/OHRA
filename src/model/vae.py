import os
import sys
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from termcolor import colored
from torch.nn import functional as F

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        # ---------- Encoder ----------
        enc_layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_features, h_dim))
            enc_layers.append(nn.BatchNorm1d(h_dim))
            enc_layers.append(nn.ReLU())
            in_features = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        # ---------- Decoder ----------
        dec_layers = []
        in_features = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_features, h_dim))
            dec_layers.append(nn.BatchNorm1d(h_dim))
            dec_layers.append(nn.ReLU())
            in_features = h_dim
        self.decoder = nn.Sequential(*dec_layers)

        self.fc_out_cont = nn.Linear(in_features, 32)
        self.fc_out_axis1 = nn.Linear(in_features, 6)
        self.fc_out_axis2 = nn.Linear(in_features, 6)
        self.fc_out_disc = nn.Linear(in_features, 22)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        recon_cont = self.fc_out_cont(h)
        recon_axis1 = self.fc_out_axis1(h)
        recon_axis2 = self.fc_out_axis2(h)
        recon_disc = torch.sigmoid(self.fc_out_disc(h))
        recon = torch.cat([recon_cont, recon_axis1, recon_axis2, recon_disc], dim=1)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = VAE(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            latent_dim=cfg.latent_dim
        )

    def training_step(self, batch, batch_idx):
        recon_batch, mu, logvar = self.model(batch)

        recon_loss_cont = F.mse_loss(recon_batch[:, :32], batch[:, :32], reduction='mean')
        recon_loss_axis1 = F.cross_entropy(recon_batch[:, 32:38], torch.argmax(batch[:, 32:38], dim=1), reduction='mean')
        recon_loss_axis2 = F.cross_entropy(recon_batch[:, 38:44], torch.argmax(batch[:, 38:44], dim=1), reduction='mean')
        recon_loss_disc = F.binary_cross_entropy(recon_batch[:, 44:], batch[:, 44:], reduction='mean')
        recon_loss = recon_loss_cont + recon_loss_axis1 + recon_loss_axis2 + recon_loss_disc

        kl_beta = self.cfg.kl_beta
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = recon_loss + kl_beta * kl_loss

        self.log("loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("recon_loss_cont", recon_loss_cont, prog_bar=True)
        self.log("recon_loss_axis1", recon_loss_axis1, prog_bar=True)
        self.log("recon_loss_axis2", recon_loss_axis2, prog_bar=True)
        self.log("recon_loss_disc", recon_loss_disc, prog_bar=True)
        self.log("kl_loss", kl_beta * kl_loss, prog_bar=True)
        self.log("kl_beta", kl_beta, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer
