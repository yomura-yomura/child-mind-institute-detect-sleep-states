import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path

import hydra
import wandb
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import SegDataModule
from cmi_dss_lib.modelmodule.seg import SegModel
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from omegaconf import OmegaConf

from child_mind_institute_detect_sleep_states.model.callbacks import ModelCheckpointWithSymlinkToBest
from child_mind_institute_detect_sleep_states.model.loggers import WandbLogger

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = ["config/omura/v100/1d.yaml"]
else:
    args = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


project_root_path = pathlib.Path(__file__).parent.parent


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    print(cfg)

    seed_everything(cfg.seed)

    # init lightning model
    datamodule = SegDataModule(cfg)
    LOGGER.info("Set Up DataModule")

    model = SegModel(
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )

    model_save_dir_path = project_root_path / cfg.dir.output_dir / "train" / cfg.exp_name / cfg.split.name

    trainer = Trainer(
        devices=1,
        # env
        default_root_dir=model_save_dir_path,
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.epoch,
        max_steps=cfg.epoch * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[
            ModelCheckpointWithSymlinkToBest(
                dirpath=model_save_dir_path,
                filename="{epoch}-{EventDetectionAP:.3f}",
                verbose=True,
                monitor=cfg.monitor,
                mode=cfg.monitor_mode,
                save_top_k=2,
                save_last=True,
            ),
            EarlyStopping(
                monitor=cfg.monitor,
                mode=cfg.monitor_mode,
                patience=cfg.early_stopping_patience,
            ),
            LearningRateMonitor("epoch"),
            # RichProgressBar(),
            # RichModelSummary(max_depth=2),
        ],
        logger=WandbLogger(
            name=f"{cfg.exp_name}-{cfg.split.name}",
            project="child-mind-institute-detect-sleep-states",
            group=cfg.exp_name,
        ),
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="+")
    parser.add_argument("--folds", type=str, default=None)
    args = parser.parse_args(args)

    if args.folds is None:
        folds = list(range(5))
    else:
        folds = list(map(int, args.folds.split(",")))

    print(f"{folds = }")

    for i_fold in folds:
        overrides_args = []
        for p in args.config_path:
            overrides_args += OmegaConf.load(project_root_path / p)
        overrides_args.append(f"split=fold_{i_fold}")
        print(f"{overrides_args = }")
        sys.argv = sys.argv[:1] + overrides_args
        main()
