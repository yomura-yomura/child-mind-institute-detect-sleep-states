import dataclasses
import pathlib

import lightning as L
import numpy as np
import omegaconf
import polars as pl
import torch
import torch.utils.data
from nptyping import NDArray

from ..config.train import DirConfig, Model, TrainSplit, TrainSplitType
from .seg import TrainDataset

project_root_path = pathlib.Path(__file__).parent.parent.parent


@dataclasses.dataclass
class StackingConfig:
    dir: DirConfig

    split: "TrainSplit"
    split_type: "TrainSplitType"

    seed: int
    exp_name: str

    duration: int
    prev_margin_steps: int
    next_margin_steps: int

    downsample_rate: int
    upsample_rate: int

    input_model_names: list[str]

    model: Model

    duration: int


class StackingDataModule(L.LightningDataModule):
    def __init__(self, cfg: StackingConfig):
        super().__init__()

        self.cfg = cfg
        self.data_dir = pathlib.Path(cfg.dir.data_dir)
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()

        with open(
            project_root_path
            / "run"
            / "conf"
            / "split"
            / self.cfg.split_type.name
            / f"{self.cfg.split.name}.yaml"
        ) as f:
            series_ids_dict = omegaconf.OmegaConf.load(f)
        self.train_series_ids = series_ids_dict["train_series_ids"]
        self.valid_series_ids = series_ids_dict["valid_series_ids"]
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.train_series_ids)
        )
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.valid_series_ids)
        )

        self.train_features = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_predicted_paths = [
                pathlib.Path(
                    self.cfg.dir.sub_dir,
                    "predicted",
                    input_model_name,
                    "dev",
                    f"{self.cfg.split.name}",
                )
                for input_model_name in self.cfg.input_model_names
            ]
            self.train_features = load_features(train_predicted_paths, self.train_series_ids)

        if stage in ("fit", "valid"):
            valid_predicted_paths = [
                pathlib.Path(
                    self.cfg.dir.sub_dir,
                    "predicted",
                    input_model_name,
                    "train",
                    f"{self.cfg.split.name}",
                )
                for input_model_name in self.cfg.input_model_names
            ]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )


def load_features(
    predicted_paths: list[pathlib.Path | str],
    series_ids: list[str],
) -> dict[str, NDArray]:
    return {
        series_id: np.stack(
            [
                np.load(predicted_path / f"{series_id}.npz")["arr_0"]
                for predicted_path in predicted_paths
            ],
            axis=-1,
        )
        for series_id in series_ids
    }
