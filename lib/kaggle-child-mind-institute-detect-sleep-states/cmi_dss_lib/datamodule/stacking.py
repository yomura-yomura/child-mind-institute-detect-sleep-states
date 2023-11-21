import pathlib

import lightning as L
import numpy as np
import omegaconf
import polars as pl
import torch
import torch.utils.data
from nptyping import Float, NDArray, Shape

from ..config import StackingConfig
from .seg import Indexer, TestDataset, TrainDataset, ValidDataset, pad_if_needed

project_root_path = pathlib.Path(__file__).parent.parent.parent


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
        self.valid_chunk_features = None
        self.test_chunk_features = None

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
            self.train_features = load_features(
                train_predicted_paths, series_ids=self.train_series_ids
            )

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
            self.valid_chunk_features = load_chunk_features(
                valid_predicted_paths,
                series_ids=self.valid_series_ids,
                duration=self.cfg.duration,
                prev_margin_steps=self.cfg.prev_margin_steps,
                next_margin_steps=self.cfg.next_margin_steps,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
            num_features=len(self.cfg.input_model_names),
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            event_df=self.valid_event_df,
            chunk_features=self.valid_chunk_features,
            num_features=len(self.cfg.input_model_names),
        )
        return torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


def load_features(
    predicted_paths: list[pathlib.Path | str],
    series_ids: list[str],
) -> dict[str, NDArray[Shape["3, *, *"], Float]]:
    return {
        series_id: np.stack(
            [
                np.load(predicted_path / f"{series_id}.npz")["arr_0"].T
                for predicted_path in predicted_paths
            ],
            axis=-1,
        )  # (pred_type, duration, model)
        for series_id in series_ids
    }


def load_chunk_features(
    predicted_paths: list[pathlib.Path | str],
    series_ids: list[str],
    duration: int,
    prev_margin_steps: int = 0,
    next_margin_steps: int = 0,
) -> dict[str, NDArray[Shape["3, *, *"], Float]]:
    chunk_features = {}

    for series_id in series_ids:
        this_feature = np.stack(
            [
                np.load(predicted_path / f"{series_id}.npz")["arr_0"].T
                for predicted_path in predicted_paths
            ],
            axis=-1,
        )  # (pred_type, duration, model)

        indexer = Indexer(this_feature.shape[-2], duration, prev_margin_steps, next_margin_steps)

        num_chunks = (this_feature.shape[-2] // indexer.interest_duration) + 1
        for i in range(num_chunks):
            key = f"{series_id}_{i:07}"

            start, end = indexer.get_interest_range(i)

            mask = np.zeros(this_feature.shape[-2], dtype=bool)
            mask[start:end] = True

            # extend crop area with margins
            start, end = indexer.get_cropping_range(i)

            chunk_features[f"{key}_mask"] = pad_if_needed(mask[start:end], duration, pad_value=0)
            chunk_features[key] = pad_if_needed(
                this_feature[..., start:end, :], duration, pad_value=0, axis=-2
            )

    return chunk_features