import pathlib
import sys
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ...data.comp_dataset import event_mapping

Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor, np.str_, torch.Tensor]

project_root_path = pathlib.Path(__file__).parent.parent.parent.parent

__all__ = ["Batch", "Dataset", "UserWiseDataset", "TimeWindowDataset", "get_dataset", "get_data", "preprocess"]

sys.path.append(str(project_root_path / "lib"))

from sleep_stage_classification.code.LSTM import Dataset

sys.path.pop(-1)


target_columns = [f"event_{k}" for k in event_mapping]


def preprocess(
    df_: pd.DataFrame,
    prev_steps_in_epoch: int,
    n_prev_time: int,
    *,
    next_steps_in_epoch: int = 1,
    n_interval_steps: int = 1,
    remove_nan: bool = True,
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.int_]]:
    indices = (
        np.arange(len(df_) - prev_steps_in_epoch - next_steps_in_epoch)[:, np.newaxis]
        + np.arange(prev_steps_in_epoch + next_steps_in_epoch)[np.newaxis, :]
    )

    enmo, angle = (
        np.swapaxes(
            np.take(
                pd.concat([df_[key].shift(-i * n_interval_steps) for i in range(n_prev_time)], axis=1).to_numpy(),
                indices,
                axis=0,
            ),
            axis1=1,
            axis2=2,
        )  # records, n_prev_time, steps_in_epochs
        for key in ["enmo", "anglez"]
    )

    # selected_event_df = reference_df_[reference_df_["series_id"].isin([series_id])]
    labels_ = df_[target_columns].to_numpy()[indices[:, -next_steps_in_epoch]]

    if remove_nan:
        sel = ~np.any(np.isnan(enmo), axis=(1, 2))
        enmo = enmo[sel]
        labels_ = labels_[sel]

    return enmo, angle, labels_


def get_dataset(
    df: pd.DataFrame,
    device: str | torch.device,
    *,
    prev_steps_in_epoch: int,
    next_steps_in_epoch: int,
    n_prev_time: int,
    n_interval_steps: int = 1,
):
    return Dataset(
        *get_data(
            df,
            prev_steps_in_epoch=prev_steps_in_epoch,
            next_steps_in_epoch=next_steps_in_epoch,
            n_prev_time=n_prev_time,
            n_interval_steps=n_interval_steps,
        ),
        device=device,
    )


def get_data(
    df: pd.DataFrame, *, prev_steps_in_epoch: int, next_steps_in_epoch: int, n_prev_time: int, n_interval_steps: int = 1
) -> tuple[NDArray[np.float_], NDArray[np.int_], NDArray[np.str_]]:
    series_id_list = []
    data_list = []
    label_list = []
    for series_id, step_df in tqdm(df.groupby("series_id")):  # noqa
        enmo, angle, labels = preprocess(
            step_df,
            prev_steps_in_epoch=prev_steps_in_epoch,
            next_steps_in_epoch=next_steps_in_epoch,
            n_prev_time=n_prev_time,
            n_interval_steps=n_interval_steps,
            remove_nan=True,
        )
        if len(enmo) == 0:
            continue
        enmo = enmo[:, -1, :][..., np.newaxis, :]
        angle = angle[:, -1, :][..., np.newaxis, :]

        data = np.concatenate([enmo, angle], axis=-2)

        n_records = len(enmo)
        assert n_records == len(enmo) == len(labels)
        series_id_list.append(np.array([series_id] * n_records, dtype=str))
        data_list.append(data)
        label_list.append(labels)
    return (
        np.concatenate(data_list, axis=0),
        np.concatenate(label_list, axis=0),
        np.concatenate(series_id_list, axis=0),
    )


import os

import numpy.lib.recfunctions
import polars as pl
from numpy.typing import NDArray


def reshape(a, new_shape, drop=True) -> NDArray:
    from numpy_utility import is_array, is_integer

    if is_array(new_shape):
        pass
    elif is_integer(new_shape):
        new_shape = (new_shape,)
    else:
        raise TypeError(f"'{type(new_shape)}' object cannot be interpreted as an integer")

    a = np.array(a)
    new_shape = np.array(new_shape)

    i_unknown_dimensions = np.where(new_shape < 0)[0]
    if i_unknown_dimensions.size == 0:
        pass
    elif i_unknown_dimensions.size == 1:
        new_shape[i_unknown_dimensions[0]] = np.floor(a.size / np.prod(new_shape[new_shape >= 0]))
    else:
        raise ValueError("can only specify one unknown dimension")

    new_size = a.size - a.size % np.prod(new_shape)
    if a.size < new_size:
        raise ValueError(f"cannot reshape array of size {a.size} into shape {tuple(new_shape)}")

    if drop is True:
        return a.flatten()[:new_size].reshape(new_shape)
    else:
        return a.reshape(new_shape)


class UserWiseDataset(Dataset):
    def __init__(
        self,
        df: pl.LazyFrame,
        in_memory: bool = True,
    ):
        self.unique_series_ids = (
            df.select(pl.col("series_id").unique(maintain_order=True)).collect().to_numpy().flatten()
        )
        self.df = df.select(pl.col(["series_id", "step", "enmo", "anglez", *target_columns]))
        self.in_memory = in_memory

        if self.in_memory:
            self.df = self.df.collect()

    def __len__(self):
        return len(self.unique_series_ids)

    def __getitem__(self, idx: int) -> Batch:
        target_series_id = self.unique_series_ids[idx]
        target_df = self.df.filter(pl.col("series_id") == target_series_id).sort("step")
        if not self.in_memory:
            target_df = target_df.collect()

        features = target_df[["enmo", "anglez"]].to_numpy()
        labels = target_df[target_columns].to_numpy().astype(np.float32)
        steps = target_df["step"].to_numpy().astype(np.int64)

        # round to 1min

        features = reshape(features, (-1, 12, 2))
        features = np.stack(
            [
                np.mean(features, axis=1),
                np.std(features, axis=1),
                np.median(features, axis=1),
                np.max(features, axis=1),
                np.min(features, axis=1),
            ],
            axis=2,
        ).reshape(
            len(features), -1
        )  # (step, feature)

        labels = reshape(labels, (-1, 12, 3))
        labels = np.mean(labels, axis=1)
        labels = labels[:, 1:]  # only wakeup and onset

        steps = reshape(steps, (-1, 12))[:, 12 // 2]

        labels = (labels - np.mean(labels, axis=0, keepdims=True)) / (np.std(labels, axis=0, keepdims=True) + 1e-16)

        return torch.from_numpy(features), torch.from_numpy(labels), target_series_id, torch.from_numpy(steps)


class TimeWindowDataset(Dataset):
    def __init__(
        self,
        df: pl.LazyFrame,
        *,
        prev_steps_in_epoch: int,
        total_steps_in_epoch: int,
        device: str | torch.device,
        cache_dir: str | os.PathLike[str] | None = None,
    ):
        self.device = device
        self.unique_series_ids = (
            df.select("series_id").unique(maintain_order=True).collect().to_numpy().astype(str).flatten()
        )
        self.total_steps_in_epoch = total_steps_in_epoch

        counts = df.group_by("series_id", maintain_order=True).count().select("count").collect()
        counts = (
            (np.ceil((counts + prev_steps_in_epoch) / total_steps_in_epoch) * total_steps_in_epoch)
            .astype(int)
            .flatten()
        )

        if cache_dir is not None and os.path.exists(cache_dir):
            data = np.load(cache_dir)["arr_0"]
        else:
            n_total_records = np.sum(counts)

            data = np.zeros(
                n_total_records,
                dtype=[
                    ("series_id", np.uint16),
                    ("step", np.uint32),
                    ("enmo", np.float32),
                    ("anglez", np.float32),
                    # ("1min_ma_enmo", np.float32), ("1min_ma_anglez", np.float32)
                    *((col, np.float32) for col in target_columns),
                ],
            )
            print(f"{data.nbytes / 1024 ** 3 = :.2f} GB")

            cursor = 0
            for i, series_id in enumerate(tqdm(self.unique_series_ids, desc="creating dataset")):
                df_ = (
                    df.filter(pl.col("series_id") == series_id)
                    .select(
                        pl.col("step").cast(pl.UInt32),
                        pl.col("enmo").cast(pl.Float32),
                        pl.col("anglez").cast(pl.Float32),
                        *(pl.col(col).cast(pl.Float32) for col in target_columns),
                    )
                    .collect()
                )

                n_records = len(df_) + prev_steps_in_epoch

                if n_records % total_steps_in_epoch > 0:
                    n_records += total_steps_in_epoch - n_records % total_steps_in_epoch

                n_records_to_pad = n_records - len(df_)
                if n_records_to_pad > 0:
                    df_ = pl.concat([*([df_[0]] * n_records_to_pad), df_])
                # df_ = (
                #     df_
                #     .with_columns(
                #         pl.col("enmo").rolling_mean(window_size=12).alias("1min-ma-enmo"),
                #         pl.col("anglez").rolling_mean(window_size=12).alias("1min-ma-anglez"),
                #     )
                # )
                next_cursor = cursor + len(df_)
                data[cursor:next_cursor]["series_id"] = i
                for col in df_.columns:
                    data[cursor:next_cursor][col] = df_[col]
                cursor = next_cursor

            assert len(data) == cursor

            if cache_dir is not None:
                np.savez(cache_dir, data)

        indices = np.ma.asarray(
            np.repeat(np.arange(counts.max())[np.newaxis], len(counts), axis=0)
            + np.cumsum([0, *counts[:-1]])[:, np.newaxis]
        )
        indices.mask = np.arange(counts.max()) >= counts[:, np.newaxis] - prev_steps_in_epoch
        indices = indices.compressed()
        self.indices = indices

        sum_weight = np.array([np.take(data[col], self.indices).sum() for col in target_columns])
        print(f"{target_columns} = {sum_weight / sum_weight.sum()}")

        self.features = np.lib.recfunctions.structured_to_unstructured(data[["enmo", "anglez"]], copy=True)
        self.series_ids = data["series_id"]
        self.labels = np.lib.recfunctions.structured_to_unstructured(data[target_columns], copy=True)
        self.steps = data["step"].astype("i8")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Batch:
        idx = self.indices[idx]
        # selected = self.data[idx : idx + total_steps_in_epoch]
        # features = np.lib.recfunctions.structured_to_unstructured(selected[["enmo", "anglez"]], copy=True)
        interest = slice(idx, idx + self.total_steps_in_epoch)

        features = self.features[interest]

        unique_series_ids = np.unique(self.series_ids[interest])
        assert len(unique_series_ids) == 1
        series_id = unique_series_ids[0]
        uid = self.unique_series_ids[series_id]

        steps = self.steps[interest]

        labels = self.labels[interest]
        return (
            torch.from_numpy(features).float().to(self.device),
            torch.from_numpy(labels).float().to(self.device),
            uid,
            torch.from_numpy(steps).cpu(),
        )
