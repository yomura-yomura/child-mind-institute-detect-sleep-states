import os
import pathlib
from typing import TypeAlias

import numpy as np
import numpy.lib.recfunctions
import polars as pl
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm.auto import tqdm, trange

from ...data.comp_dataset import event_mapping

Batch: TypeAlias = tuple[torch.Tensor, np.str_, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, np.str_, torch.Tensor]

project_root_path = pathlib.Path(__file__).parent.parent.parent.parent

__all__ = ["Batch", "UserWiseDataset", "TimeWindowDataset"]


target_columns = [f"event_{k}" for k in event_mapping]


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
        *,
        agg_interval: int,
        feature_names: list[str],
        use_labels: bool = True,
        in_memory: bool = True,
    ):
        self.unique_series_ids = (
            df.select(pl.col("series_id").unique(maintain_order=True)).collect().to_numpy().flatten()
        )
        self.use_labels = use_labels

        common_columns = ["series_id", "step", "enmo", "anglez"]
        if self.use_labels:
            self.df = df.select(pl.col([*common_columns, *target_columns]))
        else:
            self.df = df.select(pl.col(common_columns))

        self.df_list_in_memory = None
        if in_memory:
            self.df = self.df.collect()
            self.df_list_in_memory = [
                self._get_target_df(i)[1].to_pandas()
                for i in trange(len(self.unique_series_ids), desc="moving df into memory")
            ]
            self.df = None

        self.agg_interval = agg_interval
        self.feature_names = feature_names

    def __len__(self):
        return len(self.unique_series_ids)

    def __getitem__(self, idx: int) -> Batch:
        target_series_id, target_df = self._get_target_df(idx)

        features = target_df[["enmo", "anglez"]].to_numpy()
        steps = target_df["step"].to_numpy().astype(np.int64)

        features = reshape(features, (-1, self.agg_interval, 2))

        features_list = []
        if "mean" in self.feature_names:
            features_list.append(np.mean(features, axis=1))
        if "std" in self.feature_names:
            features_list.append(np.std(features, axis=1))
        if "median" in self.feature_names:
            features_list.append(np.median(features, axis=1))
        if "max" in self.feature_names:
            features_list.append(np.max(features, axis=1))
        if "min" in self.feature_names:
            features_list.append(np.min(features, axis=1))

        features = np.stack(
            features_list,
            axis=2,
        ).reshape(
            len(features), -1
        )  # (step, feature)

        steps = reshape(steps, (-1, self.agg_interval))[:, self.agg_interval // 2]

        if self.use_labels:
            labels = target_df[target_columns].to_numpy().astype(np.float32)
            labels = reshape(labels, (-1, self.agg_interval, 3))
            labels = np.mean(labels, axis=1)
            labels = labels[:, 1:]  # only wakeup and onset
            labels = (labels - np.mean(labels, axis=0, keepdims=True)) / (np.std(labels, axis=0, keepdims=True) + 1e-16)
            return torch.from_numpy(features), target_series_id, torch.from_numpy(steps), torch.from_numpy(labels)
        else:
            return torch.from_numpy(features), target_series_id, torch.from_numpy(steps)

    def _get_target_df(self, idx) -> tuple[np.str_, pl.DataFrame]:
        target_series_id = self.unique_series_ids[idx]
        if self.df_list_in_memory is None:
            target_df = self.df.filter(pl.col("series_id") == target_series_id).sort("step")
            if isinstance(target_df, pl.LazyFrame):
                target_df = target_df.collect()
        else:
            target_df = self.df_list_in_memory[idx]
        return target_series_id, target_df


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
