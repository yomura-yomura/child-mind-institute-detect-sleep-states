import argparse
import logging
import os
import pathlib
from pathlib import Path

import hydra
import wandb
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import SegDataModule
from cmi_dss_lib.modelmodule.seg import SegChunkModule
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
import cmi_dss_lib.utils.hydra

from child_mind_institute_detect_sleep_states.model.callbacks import EarlyStopping
from child_mind_institute_detect_sleep_states.model.loggers import WandbLogger

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        "../config/exp/50_resume.yaml",
        # "--folds", "1,2,3,4"
    ]
else:
    args = None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


project_root_path = pathlib.Path(__file__).parent.parent


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    print(cfg)

    seed_everything(cfg.seed)

    # init lightning model
    datamodule = SegDataModule(cfg)
    LOGGER.info("Set Up DataModule")

    model_save_dir_path = (
        project_root_path / cfg.dir.output_dir / "train" / cfg.exp_name / cfg.split.name
    )

    model = SegChunkModule(
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
        model_save_dir_path=model_save_dir_path,
    )

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
        # limit_val_batches=0.3,
        # limit_val_batches=0.0 if cfg.val_after_steps > 0 else 1.0,
        callbacks=[
            # ModelCheckpoint(
            #     dirpath=model_save_dir_path,
            #     filename="{epoch}-{step}-{EventDetectionAP:.3f}",
            #     verbose=True,
            #     monitor=cfg.monitor,
            #     mode=cfg.monitor_mode,
            #     save_top_k=3,
            #     save_last=True,
            #     every_n_train_steps=cfg.val_check_interval,
            #     # val_after_steps=cfg.val_after_steps,
            # ),
            EarlyStopping(
                monitor=cfg.monitor,
                mode=cfg.monitor_mode,
                patience=cfg.early_stopping_patience,
                val_after_steps=cfg.val_after_steps,
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
        val_check_interval=cfg.val_check_interval,
    )

    if cfg.resume_from_checkpoint is None:
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = os.path.join(
            cfg.resume_from_checkpoint, cfg.split.name, "last.ckpt"
        )
        if not os.path.exists(resume_from_checkpoint):
            raise FileNotFoundError(resume_from_checkpoint)
        # ckpt = torch.load(resume_from_checkpoint)
        # early_stopping_key_in_callbacks = [
        #     k for k, _ in ckpt["callbacks"].items() if k.startswith("EarlyStopping")
        # ][0]
        # ckpt["callbacks"][early_stopping_key_in_callbacks][
        #     "patience"
        # ] = cfg.early_stopping_patience
        print(f"Info: Training resumes from {resume_from_checkpoint}")
        # torch.save(ckpt, model_save_dir_path / "resume_from_checkpoint.ckpt")
        # resume_from_checkpoint = model_save_dir_path / "resume_from_checkpoint.ckpt"

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path_or_hydra_arguments", nargs="+")
    parser.add_argument("--folds", type=str, default=None)
    args = parser.parse_args(args)

    if args.folds is None:
        folds = list(range(5))
    else:
        folds = list(map(int, args.folds.split(",")))

    print(f"{folds = }")

    for i_fold in folds:
        cmi_dss_lib.utils.hydra.override_default_hydra_config(
            args.config_path_or_hydra_arguments, {"split": f"fold_{i_fold}"}
        )
        main()
