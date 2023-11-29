import copy
import pathlib
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import omegaconf
import pandas as pd
import polars as pl
import torch
from cmi_dss_lib.utils.load_predicted import load_predicted

# = result_pathfrom cmi_dss_lib.utils.post_process import post_process_for_seg
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
            this_feature.append(
                np.load(series_dir / f"{feature_name}.npy")
                if (series_dir / f"{feature_name}.npy").exists()
                else np.load(series_dir / f"{feature_name}.npz")["arr_0"]
            )
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: list[str],
    processed_dir: Path,
    phase: str,
    scale_type: str,
    inference_step_offset: int,
    prev_margin_steps: int = 0,
    next_margin_steps: int = 0,
) -> dict[str, np.ndarray]:
    features = {}

    phase_dir_path = processed_dir / phase / scale_type
    if not phase_dir_path.exists():
        raise FileNotFoundError(f"{phase_dir_path.resolve()}")

    for series_id in series_ids:
        series_dir = phase_dir_path / series_id

        this_feature = np.stack(
            [
                np.load(series_dir / f"{feature_name}.npy")
                if (series_dir / f"{feature_name}.npy").exists()
                else np.load(series_dir / f"{feature_name}.npz")["arr_0"]
                for feature_name in feature_names
            ],
            axis=1,
        )  # (duration, feature)

        if inference_step_offset is not None:
            this_feature = this_feature[inference_step_offset:]

        indexer = Indexer(this_feature.shape[0], duration, prev_margin_steps, next_margin_steps)

        num_chunks = (len(this_feature) // indexer.interest_duration) + 1
        for i in range(num_chunks):
            key = f"{series_id}_{i:07}"

            start, end = indexer.get_interest_range(i)

            mask = np.zeros(this_feature.shape[0], dtype=bool)
            mask[start:end] = True

            # extend crop area with margins
            start, end = indexer.get_cropping_range(i)

            features[f"{key}_mask"] = pad_if_needed(mask[start:end], duration, pad_value=0)

            chunk_feature = pad_if_needed(this_feature[start:end], duration, pad_value=0)
            features[key] = chunk_feature

    return features  # type: ignore


class Indexer:
    def __init__(
        self, total_duration: int, duration: int, prev_margin_steps: int, next_margin_steps: int
    ):
        self.total_duration = total_duration
        self.interest_duration = duration - prev_margin_steps - next_margin_steps
        self.duration = duration
        self.prev_margin_steps = prev_margin_steps
        self.next_margin_steps = next_margin_steps

    def get_interest_range(self, i: int) -> tuple[int, int]:
        start = i * self.interest_duration
        end = (i + 1) * self.interest_duration
        return start, end

    def get_cropping_range(self, i: int) -> tuple[int, int]:
        start, end = self.get_interest_range(i)
        start -= self.prev_margin_steps
        end += self.next_margin_steps
        if start < 0:
            end -= start
            start = 0
        elif end > self.total_duration + self.duration:
            start += end - self.total_duration
            end = self.total_duration - 1
        assert end - start == self.duration, (start, end, self.duration)
        return start, end


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), max(min(pos, max_end - duration), 0))
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(
    this_event_df: pd.DataFrame,
    labels: list[str],
    num_frames: int,
    duration: int,
    start: int,
    end: int,
) -> np.ndarray:
    assert start <= end
    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    # if "event_onset" in labels:
    #     this_event_df = this_event_df[start <= this_event_df["wakeup"]]
    # if "event_wakeup" in labels:
    #     this_event_df = this_event_df[this_event_df["onset"] <= end]
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, len(labels)))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if "event_onset" in labels and 0 <= onset < num_frames:
            label[onset, labels.index("event_onset")] = 1
        if "event_wakeup" in labels and num_frames > wakeup >= 0:
            label[wakeup, labels.index("event_wakeup")] = 1

        if "sleep" in labels:
            onset = max(0, onset)
            wakeup = min(num_frames, wakeup)
            label[onset:wakeup, labels.index("sleep")] = 1  # sleep

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


mapping = {"event_onset": "onset", "event_wakeup": "wakeup"}


def negative_sampling(this_event_df: pd.DataFrame, labels: list[str], num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    labels = [mapping[label] for label in labels if label in mapping]
    # onset/wakeupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[labels].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# Pseudo label
###################


def get_labeling_interval(
    this_event_df: pd.DataFrame,
    max_steps: int,
    th_nan_label: int = 24 * 60 * 12,
    buffer: int = 8 * 60 * 12,
):
    this_event_df["shift_onset"] = this_event_df["onset"].shift(-1).fillna(max_steps)
    pseudo_interval = this_event_df[
        this_event_df["shift_onset"] - this_event_df["wakeup"] >= th_nan_label
    ][["wakeup", "shift_onset"]].values
    first_onset = this_event_df.head(1)["onset"].values[0]
    if first_onset >= th_nan_label:
        pseudo_interval = np.concatenate(
            [np.array([[-buffer, first_onset + buffer]]), pseudo_interval]
        )
    pseudo_interval[:, 0] += buffer
    pseudo_interval[:, 1] -= buffer
    return Pseudo_interval


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
        num_features: int,
    ):
        self.cfg = cfg
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )
        self.use_pseudo_label = self.cfg.pseudo_label.use_pseudo
        if self.use_pseudo_label:
            self.pseudo_version = self.cfg.pseudo_label.use_version

            # Pseudo label version 0
            if self.pseudo_version == 0:
                self.id_to_pseudo_label = load_predicted(self.cfg.pseudo_label.v0.path_pseudo)
                self.id_to_label_interval = {}
                for s_id, df in self.event_df.groupby("series_id"):
                    self.id_to_label_interval[s_id] = (
                        get_labeling_interval(
                            this_event_df=df.reset_index(drop=True),
                            max_steps=self.id_to_pseudo_label[s_id].shape[0],
                        )
                        // self.cfg.downsample_rate
                    )

            # Pseudo label version 1
            elif self.pseudo_version == 1:
                self.df_pseudo_label = pd.read_csv(self.cfg.pseudo_label.v1.path_pseudo)

        self.available_target_labels = [
            mapping[label] for label in self.cfg.labels if label in mapping
        ]

        self.available_target_labels = [
            mapping[label] for label in self.cfg.labels if label in mapping
        ]

        if self.cfg.sampling_with_start_timing_hour:
            assert self.cfg.duration == 17280
            assert self.cfg.prev_margin_steps == self.cfg.next_margin_steps == 0
            from run.anl_start_timing_for_each_series_id import get_kde, get_start_timing_hour_dict

            self.start_timing_hour_dict = get_start_timing_hour_dict()
            self.start_timing_hour_sampler = get_kde()
        else:
            self.start_timing_hour_dict = None
            self.start_timing_hour_sampler = None

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        event = np.random.choice(
            # self.available_target_labels,
            # p=[1 / len(self.available_target_labels)] * len(self.available_target_labels),
            ["onset", "wakeup"],
            p=[0.5, 0.5],
        )
        series_id = self.event_df.at[idx, "series_id"]
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)

        # extract data matching series_id
        this_feature = self.features[series_id]  # (..., n_steps, num_features)

        n_steps = this_feature.shape[-2]

        if random.random() < self.cfg.bg_sampling_rate:
            # sample background (potentially include labels in duration)
            pos = negative_sampling(this_event_df, self.cfg.labels, n_steps)
        else:
            # always include labels in duration
            pos = self.event_df.at[idx, event]

        # crop
        if self.cfg.sampling_with_start_timing_hour:
            this_start_hour = self.start_timing_hour_dict[series_id]
            sampled_start_hour = self.start_timing_hour_sampler.resample(size=1)[0][0]

            pos_to_sample = np.arange(pos - self.cfg.duration, pos) + 1
            hours_to_sample = (this_start_hour + pos_to_sample / (12 * 60)) % 24

            order = np.argsort(hours_to_sample)
            start = int(
                pos_to_sample[
                    order[
                        min(
                            np.searchsorted(hours_to_sample[order], sampled_start_hour),
                            len(hours_to_sample) - 1,
                        )
                    ]
                ]
            )
            if start < 0:
                warnings.warn("pos < 0", UserWarning)
                start = max(start, 0)
            if start > n_steps - self.cfg.duration:
                warnings.warn("pos > n_steps - duration", UserWarning)
                start = min(start, n_steps - self.cfg.duration)
            end = start + self.cfg.duration
        else:
            start, end = random_crop(
                pos,
                self.cfg.duration,
                n_steps,
            )

        feature = this_feature[..., start:end, :]  # (..., duration, num_features)

        # upsample
        feature = torch.FloatTensor(feature.swapaxes(-2, -1)).unsqueeze(0)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        if not np.isfinite(feature).all():
            raise RuntimeError(f"encountered nan/inf in feature: {feature}")

        # from hard label to gaussian label
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_label(
            this_event_df, list(self.cfg.labels), num_frames, self.cfg.duration, start, end
        )

        # 正解ラベル周辺の予測値を使った疑似ラベリング
        if self.use_pseudo_label and self.pseudo_version == 1:
            label = self.pseudo_labeling(
                label=label, start=start, end=end, num_frames=num_frames, series_id=series_id
            )

        target_indices = []
        if "event_onset" in self.cfg.labels:
            i = list(self.cfg.labels).index("event_onset")
            label[:, [i]] = gaussian_label(
                label[:, [i]],
                offset=self.cfg.offset_onset or self.cfg.offset,
                sigma=self.cfg.sigma_onset or self.cfg.sigma,
            )
        if "event_wakeup" in self.cfg.labels:
            i = list(self.cfg.labels).index("event_wakeup")
            label[:, [i]] = gaussian_label(
                label[:, [i]],
                offset=self.cfg.offset_wakeup or self.cfg.offset,
                sigma=self.cfg.sigma_wakeup or self.cfg.sigma,
            )
        if not np.isfinite(label).all():
            raise RuntimeError(f"encountered nan/inf in label: {label}")
            target_indices.append(list(self.cfg.labels).index("event_wakeup"))

        # if not self.use_pseudo_label and not self.pseudo_version == 1:
        # label[:, target_indices] = gaussian_label(
        #    label[:, target_indices], offset=self.cfg.offset, sigma=self.cfg.sigma
        # )

        # ラベルがない区間の疑似ラベリング
        if self.use_pseudo_label and self.pseudo_version == 0:
            label = self.pseudo_labeling(
                label=label, start=start, end=end, num_frames=num_frames, series_id=series_id
            )

        return {
            "series_id": series_id,
            "feature": feature,  # (..., num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
        }

    def pseudo_labeling(self, label, start: int, end: int, num_frames: int, series_id: str):
        """Pseudo labeling"""

        # ver0 ラベルがない場所に疑似ラベリング

        if self.pseudo_version == 0:
            th_sleep = self.cfg.pseudo_label.v0.th_sleep
            th_prop = self.cfg.pseudo_label.v0.th_prop
            pseudo_idx = np.arange(
                start // self.cfg.downsample_rate, end // self.cfg.downsample_rate
            )
            is_pseudo = False
            for wakeup, onset in self.id_to_label_interval[series_id]:
                mask = (pseudo_idx >= wakeup) & (pseudo_idx <= onset)
                if np.any(mask):
                    is_pseudo = True
                    break

            if is_pseudo:
                pseudo_idx = np.arange(num_frames)[mask]

                # Pseudo labelを呼び出し
                pseudo_label = self.id_to_pseudo_label[series_id].reshape(-1, 3)[start:end]

                # down sampling with mean
                # pseudo_label = pseudo_label.reshape(num_frames,3,-1).mean(axis = 2)
                pseudo_label = (
                    resize(
                        torch.from_numpy(pseudo_label).unsqueeze(0),
                        size=[num_frames, 3],
                        antialias=False,
                    )
                    .squeeze(0)
                    .numpy()
                )
                pseudo_label = pseudo_label[pseudo_idx]

                # crop sleep
                pseudo_label[:, 0] = np.where(
                    pseudo_label[:, 0] <= th_sleep, 0, pseudo_label[:, 0]
                )

                # crop wakeup and onset prop
                pseudo_label[:, 1] = np.where(pseudo_label[:, 1] <= th_prop, 0, pseudo_label[:, 1])
                pseudo_label[:, 2] = np.where(pseudo_label[:, 2] <= th_prop, 0, pseudo_label[:, 2])
                # ori_lab = copy.deepcopy(label)

                label[pseudo_idx] = pseudo_label
        elif self.pseudo_version == 1:
            watch_interval = 12 * 60 * self.cfg.pseudo_label.v1.watch_interval
            th_min_interval = 12 * 10  # 10min
            df_pseudo = self.df_pseudo_label.query(
                "series_id == @series_id & step <= @end & step >= @start"
            )
            # save pseudo label
            if self.cfg.pseudo_label.save_pseudo:
                did_labeling = False
                ori_label = copy.deepcopy(label)

            if not df_pseudo.empty:
                labels = list(self.cfg.labels)
                for idx, event, score in df_pseudo[["step", "event", "score"]].values:
                    # idxのwatch_interval間に正解ラベルが存在しないか判定
                    not_exist_true_label = self.event_df[event][
                        idx - watch_interval : idx + watch_interval
                    ].empty

                    # 周辺に正解ラベルがない場合は早期リターン
                    if not_exist_true_label:
                        continue
                    did_labeling = True
                    # get_labels関数と処理は同じ
                    label_idx = int((idx - start) / self.cfg.duration * num_frames)
                    if (
                        event == "onset"
                        and 0 <= label_idx < num_frames
                        and "event_onset" in labels
                    ):
                        # 正解ラベルが入っている場合はスキップ
                        if (
                            np.max(
                                label[
                                    max(0, label_idx - th_min_interval) : min(
                                        label_idx + th_min_interval, num_frames
                                    ),
                                    labels.index("event_onset"),
                                ]
                            )
                            < 1
                        ):
                            label[label_idx, labels.index("event_onset")] = score
                    if (
                        event == "wakeup"
                        and num_frames > label_idx >= 0
                        and "event_wakeup" in labels
                    ):
                        # 正解ラベルが入っている場合はスキップ
                        if (
                            np.max(
                                label[
                                    max(0, label_idx - th_min_interval) : min(
                                        label_idx + th_min_interval, num_frames
                                    ),
                                    labels.index("event_wakeup"),
                                ]
                            )
                            < 1
                        ):
                            label[label_idx, labels.index("event_wakeup")] = score
            target_indices = []
            if "event_onset" in self.cfg.labels:
                target_indices.append(list(self.cfg.labels).index("event_onset"))
            if "event_wakeup" in self.cfg.labels:
                target_indices.append(list(self.cfg.labels).index("event_wakeup"))

            label[:, target_indices] = gaussian_label(
                label[:, target_indices], offset=self.cfg.offset, sigma=self.cfg.sigma
            )
            # 1を超えるラベルを補正
            # NOTE:
            label[label > 1] = 1

            # save pseudo label (疑似ラベリングの付与したときのみにファイル保存)
            if self.cfg.pseudo_label.save_pseudo:
                ori_label[:, target_indices] = gaussian_label(
                    ori_label[:, target_indices], offset=self.cfg.offset, sigma=self.cfg.sigma
                )

                # save pseudo
                if did_labeling:
                    path = Path(self.cfg.pseudo_label.save_path)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        Path(path) / f"{series_id}_{start}_{end}.npz",
                        label=ori_label,
                        pseudo=label,
                        start=np.array([start]),
                        end=np.array([end]),
                    )
        return label


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
        num_features: int,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = [key for key in chunk_features.keys() if not key.endswith("_mask")]
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.num_features = num_features
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.swapaxes(-2, -1)).unsqueeze(
            0
        )  # (1, ..., num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        # start = chunk_id * self.cfg.duration
        # end = start + self.cfg.duration
        total_duration = sum(
            feature.shape[-1]
            for key, features in self.chunk_features.items()
            if key.startswith(series_id)
        )
        start, end = Indexer(
            total_duration,
            self.cfg.duration,
            self.cfg.prev_margin_steps,
            self.cfg.next_margin_steps,
        ).get_cropping_range(chunk_id)

        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate

        label = get_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            list(self.cfg.labels),
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
    def __init__(self, cfg: TrainConfig, chunk_features: dict[str, np.ndarray], num_features: int):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = [key for key in chunk_features.keys() if not key.endswith("_mask")]
        self.num_features = num_features
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.swapaxes(-2, -1)).unsqueeze(
            0
        )  # (1, ..., num_features, duration)
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
                phase=self.cfg.phase,
                scale_type=self.cfg.scale_type,
                inference_step_offset=self.cfg.inference_step_offset,
                prev_margin_steps=self.cfg.prev_margin_steps,
                next_margin_steps=self.cfg.next_margin_steps,
            )

        series_ids = [
            series_dir.name
            for series_dir in (self.processed_dir / self.cfg.phase / self.cfg.scale_type).glob("*")
            if series_dir.is_dir() and not series_dir.name.startswith(".")
        ]
        if stage == "test":
            self.test_chunk_features = load_chunk_features(
                duration=self.cfg.duration,
                feature_names=self.cfg.features,
                series_ids=series_ids,
                processed_dir=self.processed_dir,
                phase="test",
                scale_type=self.cfg.scale_type,
                inference_step_offset=self.cfg.inference_step_offset,
                prev_margin_steps=self.cfg.prev_margin_steps,
                next_margin_steps=self.cfg.next_margin_steps,
            )

        if stage == "dev":
            self.test_chunk_features = load_chunk_features(
                duration=self.cfg.duration,
                feature_names=self.cfg.features,
                series_ids=series_ids,
                processed_dir=self.processed_dir,
                phase="dev",
                scale_type=self.cfg.scale_type,
                inference_step_offset=self.cfg.inference_step_offset,
                prev_margin_steps=self.cfg.prev_margin_steps,
                next_margin_steps=self.cfg.next_margin_steps,
            )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
            num_features=len(self.cfg.features),
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
            num_features=len(self.cfg.features),
        )
        return torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.valid_batch_size or self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        test_dataset = TestDataset(
            self.cfg,
            chunk_features=self.test_chunk_features,
            num_features=len(self.cfg.features),
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
