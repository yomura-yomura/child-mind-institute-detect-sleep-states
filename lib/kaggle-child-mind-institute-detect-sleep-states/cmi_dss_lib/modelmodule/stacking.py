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

        hidden_size = 32

        self.loss_fn = nn.BCEWithLogitsLoss()

        from ..models.feature_extractor.stacked_gru import StackedGRUFeatureExtractor

        if cfg.get("feature_extractor", None) is None:
            self.embedding_linear = nn.Linear(len(cfg.input_model_names), hidden_size)
            self.feature_extractors = None
            self.concat_linear = nn.Linear(hidden_size, 1)
        elif cfg.feature_extractor.name == "StackedGRUFeatureExtractor":
            self.embedding_linear = None
            self.feature_extractors = nn.ModuleList(
                [
                    StackedGRUFeatureExtractor(
                        in_channels=len(cfg.input_model_names),
                        out_size=cfg.duration,
                        num_layers=2,
                        hidden_size=hidden_size,
                        bidirectional=True,
                    )
                    for _ in range(3)
                ]
            )
            self.concat_linear = nn.Linear(hidden_size * 2, 1)
        else:
            raise ValueError(f"unexpected {cfg.feature_extractor=}")

    def forward(
        self, batch: dict[str : torch.Tensor], *, do_mixup=False, do_cutmix=False
    ) -> dict[str, torch.Tensor | None]:
        assert do_mixup is False and do_cutmix is False

        x = batch["feature"]
        assert x.ndim == 4  # (batch_size, pred_type, model, duration)

        if self.feature_extractors is not None:
            x = torch.stack(
                [feature_extractor(x[:, i])[:, 0] for i, feature_extractor in enumerate(self.feature_extractors)],
                dim=1,
            )  # (batch_size, pred_type, hidden_size * num_directions, duration)

        x = x.permute(0, 1, 3, 2)  # (batch_size, pred_type, duration, model)

        if self.feature_extractors is None:
            x = self.embedding_linear(x)

        x = self.model(x)  # (batch_size, pred_type, duration, model)

        logits = self.concat_linear(x).squeeze(3).permute(0, 2, 1)  # (batch_size, duration, pred_type)

        output = {"logits": logits}

        if "label" in batch.keys():
            output["loss"] = self.loss_fn(logits, batch["label"])

        return output
