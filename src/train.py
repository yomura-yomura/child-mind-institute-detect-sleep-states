import os
import pathlib

import lightning.pytorch as lp
import numpy as np
import pandas as pd
import polars as pl
import sklearn.model_selection
import torch
import torch.utils.data
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from tqdm.auto import tqdm

import child_mind_institute_detect_sleep_states as cmi_dss
import child_mind_institute_detect_sleep_states.model.sleep_stage_classification
import child_mind_institute_detect_sleep_states.score.event_detection_ap
from child_mind_institute_detect_sleep_states.data.comp_dataset import event_mapping
from child_mind_institute_detect_sleep_states.model.sleep_stage_classification.dataset import target_columns

# steps_in_epoch = 6
prev_steps_in_epoch = 12 * 20
next_steps_in_epoch = 0
# next_steps_in_epoch = 12 * 10
# next_steps_in_epoch = 12 * 30

total_steps_in_epoch = prev_steps_in_epoch + 1 + next_steps_in_epoch

# df = pd.read_parquet("sampled-corrected.parquet")
df = pl.scan_parquet("all-corrected-sigma720.parquet")
n_total_records = df.select(pl.count()).collect()[0, 0]
df = df.with_columns(index=pl.Series(np.arange(n_total_records, dtype=np.uint32)))


# indices = np.fromiter(
#     (
#         i
#         for count, cum_sum in zip(tqdm(counts), np.cumsum([0, *counts[:-1]]))
#         for i in np.arange(count - prev_steps_in_epoch) + cum_sum
#     ),
#     dtype=np.uint32,
# )


# df = df.filter(pl.col("series_id").is_in(pl.col("series_id").unique().head(30))).collect().to_pandas()


# n_prev_time = 12 * 10
# n_prev_time = 1
# n_interval_steps = prev_steps_in_epoch


# sel = label[:, 0] == 1
# sel[np.random.choice(np.where(sel)[0], size=np.count_nonzero(sel) - 5000, replace=False)] = False
# sel |= label[:, 1] > 0.1
# sel |= label[:, 2] > 0.1
# reshaped_enmo = reshaped_enmo[sel]
# label = label[sel]

import numpy.lib.recfunctions
from torch.utils.data import Dataset

from child_mind_institute_detect_sleep_states.model.sleep_stage_classification.dataset import Batch


class InMemoryDataset(Dataset):
    def __init__(
        self, df: pl.LazyFrame, *, device: str | torch.device, cache_dir: str | os.PathLike[str] | None = None
    ):
        self.device = device
        self.series_ids = df.select("series_id").unique(maintain_order=True).collect().to_numpy().astype(str).flatten()

        counts = df.group_by("series_id", maintain_order=True).count().select("count").collect()
        counts = (
            (np.ceil((counts + prev_steps_in_epoch) / total_steps_in_epoch) * total_steps_in_epoch)
            .astype(int)
            .flatten()
        )
        n_total_records = np.sum(counts)

        if cache_dir is not None and os.path.exists(cache_dir):
            data = np.load(cache_dir)["arr_0"]
        else:
            data = np.zeros(
                n_total_records,
                dtype=[
                    ("series_id", np.uint16),
                    ("enmo", np.float32),
                    ("anglez", np.float32),
                    # ("1min_ma_enmo", np.float32), ("1min_ma_anglez", np.float32)
                    *((col, np.float32) for col in target_columns),
                ],
            )
            print(f"{data.nbytes / 1024 ** 3 = :.2f} GB")

            cursor = 0
            for i, series_id in enumerate(tqdm(self.series_ids, desc="creating dataset")):
                df_ = (
                    df.filter(pl.col("series_id") == series_id)
                    .select(
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
        self.series_id = data["series_id"]
        self.labels = np.lib.recfunctions.structured_to_unstructured(data[target_columns], copy=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Batch:
        idx = self.indices[idx]
        # selected = self.data[idx : idx + total_steps_in_epoch]
        # features = np.lib.recfunctions.structured_to_unstructured(selected[["enmo", "anglez"]], copy=True)
        interest = slice(idx, idx + total_steps_in_epoch)

        features = self.features[interest]
        unique_series_ids = np.unique(self.series_id[interest])
        assert len(unique_series_ids) == 1
        series_id = unique_series_ids[0]
        uid = self.series_ids[series_id]

        labels = self.labels[interest][-1, :]
        return (
            torch.from_numpy(features.T).float().to(self.device),
            torch.from_numpy(labels).float().to(self.device),
            uid,
        )


num_epochs = 10_000

# train_batch_size = 64
# valid_batch_size = 64
# train_batch_size = 1024 * 5
train_batch_size = 1024 * 16
# train_batch_size = 512
# train_batch_size = 32
valid_batch_size = 1024 * 16
# learning_rate = 0.001 * train_batch_size / 16
learning_rate = 0.001

# weight = 1_000_000  # 0.001
# weight = 5_000_000  # 0.0005
weight = 1

n_folds = 5

data_dir_path = pathlib.Path("data")

series_id = df.select("series_id").collect().to_numpy().astype(str).flatten()
kf = sklearn.model_selection.GroupKFold(n_splits=n_folds)
for i_fold, (train_indices, valid_indices) in enumerate(kf.split(series_id, groups=series_id)):
    # train_dataset = cmi_dss.model.sleep_stage_classification.get_dataset(
    #     df.iloc[train_indices],
    #     device,
    #     prev_steps_in_epoch=prev_steps_in_epoch,
    #     next_steps_in_epoch=next_steps_in_epoch,
    #     n_prev_time=n_prev_time,
    #     n_interval_steps=n_interval_steps,
    # )
    # valid_dataset = cmi_dss.model.sleep_stage_classification.get_dataset(
    #     df.iloc[valid_indices],
    #     device,
    #     prev_steps_in_epoch=prev_steps_in_epoch,
    #     next_steps_in_epoch=next_steps_in_epoch,
    #     n_prev_time=n_prev_time,
    #     n_interval_steps=n_interval_steps,
    # )

    p = data_dir_path / f"{i_fold}" / "train.parquet"
    if p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("index").is_in(train_indices)).drop("index").collect().write_parquet(p)

    p = data_dir_path / f"{i_fold}" / "valid.parquet"
    if p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("index").is_in(valid_indices)).drop("index").collect().write_parquet(p)
    break


for i_fold in range(n_folds):
    name = f"test-#{i_fold + 1}-of-{n_folds}"
    print(name)

    p = data_dir_path / f"{i_fold}" / "train.parquet"
    train_dataset = InMemoryDataset(pl.scan_parquet(p), device="cuda", cache_dir="train_data.npz")

    p = data_dir_path / f"{i_fold}" / "valid.parquet"
    valid_dataset = InMemoryDataset(pl.scan_parquet(p), device="cuda", cache_dir="valid_data.npz")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False)

    module = cmi_dss.model.sleep_stage_classification.Module(
        steps_in_epoch=total_steps_in_epoch,
        learning_rate=learning_rate,
        wakeup_weight=weight,
        onset_weight=weight,
    )

    model_path = pathlib.Path("models")

    eval_steps = 500

    trainer = lp.Trainer(
        max_epochs=num_epochs,
        logger=WandbLogger(project="child-mind-institute-detect-sleep-states", name=name, save_dir="wandb_logs"),
        callbacks=[
            EarlyStopping(
                # monitor="valid_loss",
                # mode="min",
                monitor="EventDetectionAP",
                mode="max",
                patience=10,
            ),
            ModelCheckpoint(
                dirpath=model_path / name,
                filename="{epoch}-{valid_loss:.2f}",
                # monitor="valid_loss",
                # mode="min",
                monitor="EventDetectionAP",
                mode="max",
                save_last=True,
                save_top_k=3,
                every_n_train_steps=eval_steps,
            ),
        ],
        val_check_interval=eval_steps,
    )

    print("fitting")
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()

    break

# preds = trainer.predict(module, valid_loader)
# preds = torch.concat(preds)
# print(f"{preds = }")
# print(f"{preds.numpy().max(axis=0) = }")
