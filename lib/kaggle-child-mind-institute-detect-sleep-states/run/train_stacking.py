import argparse
import os
import pathlib

import cmi_dss_lib.utils.hydra
import hydra
import lightning as L
import wandb
from cmi_dss_lib.config import StackingConfig
from cmi_dss_lib.datamodule.stacking import StackingDataModule
from cmi_dss_lib.modelmodule.stacking import StackingChunkModule
from lightning.pytorch.callbacks import LearningRateMonitor

from child_mind_institute_detect_sleep_states.model.callbacks import EarlyStopping
from child_mind_institute_detect_sleep_states.model.loggers import WandbLogger

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        "../config/exp_for_stacking/s2.yaml",
        # "--folds", "1,2,3,4"
        "--gpus",
        "1",
    ]
else:
    args = None


project_root_path = pathlib.Path(__file__).parent.parent


@hydra.main(config_path="conf", config_name="stacking", version_base="1.2")
def main(cfg: StackingConfig):
    cfg.dir.sub_dir = str(project_root_path / "run")
    print(cfg)

    L.seed_everything(cfg.seed)

    datamodule = StackingDataModule(cfg)
    datamodule.setup("fit")
    # x = next(iter(datamodule.train_dataloader()))

    model_save_dir_path = (
        project_root_path / cfg.dir.output_dir / "train_stacking" / cfg.exp_name / cfg.split.name
    )

    module = StackingChunkModule(
        cfg, val_event_df=datamodule.valid_event_df, model_save_dir_path=model_save_dir_path
    )
    # from cmi_dss_lib.modelmodule.seg import SegChunkModule
    # module = SegChunkModule(
    #     cfg,
    #     val_event_df=datamodule.valid_event_df,
    #     model_save_dir_path=model_save_dir_path,
    #     feature_dim=len(cfg.input_model_names),
    #     num_classes=3,
    #     duration=cfg.duration,
    # )

    # preds = module.cuda()(
    #     {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    # )
    # print(preds)

    trainer = L.Trainer(
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
        callbacks=[
            EarlyStopping(
                monitor=cfg.monitor,
                mode=cfg.monitor_mode,
                patience=cfg.early_stopping_patience,
                val_after_steps=cfg.val_after_steps,
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=WandbLogger(
            name=f"{cfg.exp_name}-{cfg.split.name}",
            project="child-mind-institute-detect-sleep-states",
            group=cfg.exp_name,
        ),
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
        print(f"Info: Training resumes from {resume_from_checkpoint}")

    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_from_checkpoint)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path_or_hydra_arguments", nargs="+")
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args(args)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f'{os.environ["CUDA_VISIBLE_DEVICES"]=}')

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
        break
