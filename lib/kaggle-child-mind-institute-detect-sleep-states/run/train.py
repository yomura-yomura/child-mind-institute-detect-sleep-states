import logging
from pathlib import Path

import hydra
from cmi_dss_lib.datamodule.seg import SegDataModule
from cmi_dss_lib.modelmodule.seg import SegModel
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

import pathlib

cwd_path = pathlib.Path.cwd()


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
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

    # import torch
    #
    # module = SegModel.load_from_checkpoint(
    #     "/home/yugo.omura/child-mind-institute-detect-sleep-states/lib/kaggle-child-mind-institute-detect-sleep-states/output/train/exp002/single/best.ckpt",
    #     cfg=cfg,
    #     val_event_df=datamodule.valid_event_df,
    #     feature_dim=len(cfg.features),
    #     num_classes=len(cfg.labels),
    #     duration=cfg.duration,
    # )
    # trainer = Trainer()
    # preds = trainer.predict(module, datamodule.val_dataloader())
    # probs = [batch["logits"].sigmoid().detach().cpu() for batch in preds]
    # torch.save(module.model.state_dict(), "best_model.pth")

    from child_mind_institute_detect_sleep_states.model.callbacks import ModelCheckpointWithSymlinkToBest

    # init experiment logger
    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="child-mind-institute-detect-sleep-states-Unet",
    )

    cwd = pathlib.Path(cfg.dir.output_dir, "train", cfg.exp_name, "single")

    trainer = Trainer(
        devices=1,
        # env
        default_root_dir=cwd,
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
                dirpath=cwd,
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
                patience=20,
            ),
            LearningRateMonitor("epoch"),
            RichProgressBar(),
            RichModelSummary(max_depth=2),
        ],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)

    # # load best weights
    # model = SegModel.load_from_checkpoint(
    #     checkpoint_cb.best_model_path,
    #     cfg=cfg,
    #     val_event_df=datamodule.valid_event_df,
    #     feature_dim=len(cfg.features),
    #     num_classes=len(cfg.labels),
    #     duration=cfg.duration,
    # )
    # weights_path = "model_weights.pth"
    # LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    # torch.save(model.model.state_dict(), weights_path)

    return


if __name__ == "__main__":
    main()
