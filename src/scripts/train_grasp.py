import os
import sys
import hydra
import warnings
import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from model.grasp.pl_module import TrainingModule
from data_utils.grasp_dataloader import create_dataloader


@hydra.main(version_base="1.2", config_path="../../configs/grasp", config_name="train")
def main(cfg):
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    # pl.seed_everything(cfg.seed)

    logger = WandbLogger(
        name=cfg.name,
        save_dir=cfg.wandb.save_dir,
        project=cfg.wandb.project
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.save_dir,
        filename="{epoch}",
        every_n_epochs=cfg.training.save_every_n_epoch,
        save_top_k=-1,
        save_last=True,
        monitor=None,
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        strategy='auto',
        devices=cfg.gpu,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.max_epochs,
    )

    dataloader = create_dataloader(cfg.dataet, is_train=True)

    os.makedirs(cfg.training.save_dir, exist_ok=True)
    module = TrainingModule(cfg=cfg)
    module.train()

    trainer.fit(module, dataloader)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
