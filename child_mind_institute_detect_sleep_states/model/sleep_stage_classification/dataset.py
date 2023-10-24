import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ...data.comp_dataset import event_mapping

project_root_path = pathlib.Path(__file__).parent.parent.parent.parent

__all__ = ["Batch", "Dataset", "get_dataset", "get_data", "preprocess"]

sys.path.append(str(project_root_path / "lib"))

from sleep_stage_classification.code.LSTM import Batch, Dataset  # noqa

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
