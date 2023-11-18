import pathlib

import cmi_dss_lib.datamodule.seg
import cmi_dss_lib.models.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import polars as pl
import torch

from ..config import TrainConfig
from .base_chunk import BaseChunkModule


class SegChunkModule(BaseChunkModule):
    def __init__(
        self,
        cfg: TrainConfig,
        val_event_df: pl.DataFrame | None,
        feature_dim: int,
        num_classes: int,
        duration: int,
        model_save_dir_path: pathlib.Path | None = None,
    ):
        self.model = cmi_dss_lib.models.common.get_model(
            cfg, feature_dim=feature_dim, n_classes=num_classes, num_time_steps=self.num_time_steps
        )
        super().__init__(cfg, val_event_df, duration, model_save_dir_path)

    def forward(
        self, batch: dict[str : torch.Tensor], *, do_mixup=False, do_cutmix=False
    ) -> dict[str, torch.Tensor | None]:
        return self.model(batch["feature"], batch.get("label", None), do_mixup, do_cutmix)
