import pathlib
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from ..config import TrainConfig
from ..utils.common import pad_if_needed

project_root_path = pathlib.Path(__file__).parent.parent.parent


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    scale_type: str,
) -> dict[str, np.ndarray]:
    features = {}

    phase_dir_path = processed_dir / phase / scale_type

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in phase_dir_path.glob("*")]

    for series_id in series_ids:
        series_dir = phase_dir_path / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    scale_type: str,
    prev_margin_steps: int = 0,
    next_margin_steps: int = 0,
) -> dict[str, np.ndarray]:
    features = {}

    phase_dir_path = processed_dir / phase / scale_type
    if not phase_dir_path.exists():
        raise FileNotFoundError(f"{phase_dir_path.resolve()}")

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in phase_dir_path.glob("*")]

    for series_id in series_ids:
        series_dir = phase_dir_path / series_id

        this_feature = np.stack(
            [np.load(series_dir / f"{feature_name}.npy") for feature_name in feature_names], axis=1
        )  # (duration, feature)

        interest_duration = duration - prev_margin_steps - next_margin_steps

        num_chunks = (len(this_feature) // interest_duration) + 1
        for i in range(num_chunks):
            key = f"{series_id}_{i:07}"

            start = i * interest_duration
            end = (i + 1) * interest_duration

            mask = np.zeros(this_feature.shape[0], dtype=bool)
            mask[start:end] = True

            # extend crop area with margins
            start -= prev_margin_steps
            end += next_margin_steps
            if start < 0:
                end -= start
                start = 0
            elif end > this_feature.shape[0] + duration:
                start += end - this_feature.shape[0]
                end = this_feature.shape[0] - 1
            assert end - start == duration, (start, end, duration)

            features[f"{key}_mask"] = pad_if_needed(mask[start:end], duration, pad_value=0)

            chunk_feature = pad_if_needed(this_feature[start:end], duration, pad_value=0)
            features[key] = chunk_feature

    return features  # type: ignore


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int) -> np.ndarray:
    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if 0 <= onset < num_frames:
            label[onset, 1] = 1
        if num_frames > wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakeupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# Dataset
###################
def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        event_df: pl.DataFrame,
        features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        series_id = self.event_df.at[idx, "series_id"]
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        n_steps = this_feature.shape[0]

        if random.random() < self.cfg.bg_sampling_rate:
            # sample background (potentially include labels in duration)
            pos = negative_sampling(this_event_df, n_steps)
        else:
            # always include labels in duration
            pos = self.event_df.at[idx, event]

        # crop
        start, end = random_crop(
            pos,
            self.cfg.duration,
            n_steps,
        )

        feature = this_feature[start:end]  # (duration, num_features)

        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label[:, [1, 2]] = gaussian_label(label[:, [1, 2]], offset=self.cfg.offset, sigma=self.cfg.sigma)

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
        }


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = [key for key in chunk_features.keys() if not key.endswith("_mask")]
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "mask": self.chunk_features[f"{key}_mask"],
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = [key for key in chunk_features.keys() if not key.endswith("_mask")]
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "mask": self.chunk_features[f"{key}_mask"],
        }


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = project_root_path / cfg.dir.output_dir / "prepare_data"
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()
        self.train_series_ids = self.cfg.split.train_series_ids
        self.valid_series_ids = self.cfg.split.valid_series_ids
        self.train_event_df = self.event_df.filter(pl.col("series_id").is_in(self.train_series_ids))
        self.valid_event_df = self.event_df.filter(pl.col("series_id").is_in(self.valid_series_ids))

        self.train_features = None
        self.valid_chunk_features = None
        self.test_chunk_features = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_features = load_features(
                feature_names=self.cfg.features,
                series_ids=self.train_series_ids,
                processed_dir=self.processed_dir,
                phase="train",
                scale_type=self.cfg.scale_type,
            )
        if stage in ("fit", "valid"):
            self.valid_chunk_features = load_chunk_features(
                duration=self.cfg.duration,
                feature_names=self.cfg.features,
                series_ids=self.valid_series_ids,
                processed_dir=self.processed_dir,
                phase="train",
                scale_type=self.cfg.scale_type,
                prev_margin_steps=self.cfg.prev_margin_steps,
                next_margin_steps=self.cfg.next_margin_steps,
            )
        if stage == "test":
            series_ids = [x.name for x in (self.processed_dir / self.cfg.phase / self.cfg.scale_type).glob("*")]
            self.test_chunk_features = load_chunk_features(
                duration=self.cfg.duration,
                feature_names=self.cfg.features,
                series_ids=series_ids,
                processed_dir=self.processed_dir,
                phase="test",
                scale_type=self.cfg.scale_type,
                prev_margin_steps=self.cfg.prev_margin_steps,
                next_margin_steps=self.cfg.next_margin_steps,
            )

    def train_dataloader(self):
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

    def val_dataloader(self):
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            chunk_features=self.valid_chunk_features,
            event_df=self.valid_event_df,
        )
        return torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        test_dataset = TestDataset(self.cfg, chunk_features=self.test_chunk_features)
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
