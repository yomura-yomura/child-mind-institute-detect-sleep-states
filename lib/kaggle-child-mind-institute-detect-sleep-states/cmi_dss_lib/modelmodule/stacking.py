import pathlib

import polars as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

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
        if cfg.model.segmentation_model_name == "manet":
            self.model = smp.MAnet(
                encoder_name=cfg.model.encoder_name,
                encoder_depth=5,
                encoder_weights=cfg.model.encoder_weights,
                decoder_use_batchnorm=True,
                in_channels=3,
                classes=3,
            )
        elif cfg.model.segmentation_model_name == "unet":
            self.model = smp.Unet(
                encoder_name=cfg.model.encoder_name,
                encoder_depth=5,
                encoder_weights=cfg.model.encoder_weights,
                decoder_use_batchnorm=True,
                in_channels=3,
                classes=3,
            )
        else:
            raise ValueError(f"unexpected {cfg.model.segmentation_model_name=}")

        self.embedding_linear = nn.Linear(len(cfg.input_model_names), 32)

        self.concat_linear = nn.Linear(32, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: dict[str : torch.Tensor], *, do_mixup=False, do_cutmix=False
    ) -> dict[str, torch.Tensor | None]:
        assert do_mixup is False and do_cutmix is False

        assert batch["feature"].ndim == 4  # (batch_size, pred_type, model, duration)
        x = batch["feature"].permute(0, 1, 3, 2)  # (batch_size, pred_type, duration, model)

        x = self.embedding_linear(x)
        x = self.model(x)  # (batch_size, pred_type, duration, model)
        logits = (
            self.concat_linear(x).squeeze(3).permute(0, 2, 1)
        )  # (batch_size, duration, pred_type)

        output = {"logits": logits}

        if "label" in batch.keys():
            output["loss"] = self.loss_fn(logits, batch["label"])

        return output
