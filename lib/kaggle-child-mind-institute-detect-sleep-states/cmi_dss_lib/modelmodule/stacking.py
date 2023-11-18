import pathlib

import polars as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from ..datamodule.stacking import StackingConfig
from .base_chunk import BaseChunkModule


class StackingChunkModule(BaseChunkModule):
    def __init__(
        self,
        cfg: StackingConfig,
        val_event_df: pl.DataFrame | None,
        model_save_dir_path: pathlib.Path | None = None,
    ):
        super().__init__(cfg, val_event_df, cfg.duration, model_save_dir_path)
        self.model_sleep = smp.Unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=len(cfg.input_model_names),
            classes=1,
        )
        self.model_onset = smp.Unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=len(cfg.input_model_names),
            classes=1,
        )
        self.model_wakeup = smp.Unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=len(cfg.input_model_names),
            classes=1,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: dict[str : torch.Tensor], *, do_mixup=False, do_cutmix=False
    ) -> dict[str, torch.Tensor | None]:
        assert do_mixup is False and do_cutmix is False

        print(batch["feature"].shape)
        x = torch.concat(
            [
                self.model_sleep(batch["feature"][..., 0, :]),
                self.model_onset(batch["feature"][..., 1, :]),
                self.model_wakeup(batch["feature"][..., 2, :]),
            ],
            dim=-1,
        )

        print(x.shape)

        output = {"logits": x}

        if "label" in batch.keys():
            output["loss"] = self.loss_fn(x, batch["label"])

        return output
