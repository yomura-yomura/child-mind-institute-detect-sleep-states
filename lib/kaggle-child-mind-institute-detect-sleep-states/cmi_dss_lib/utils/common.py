import math
import os
import pathlib
import sys
import time
from contextlib import contextmanager

import numpy as np
import psutil


@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)


import collections

# should be moved to proper place
import itertools
from typing import Sequence

from numpy.typing import NDArray

import child_mind_institute_detect_sleep_states.data.comp_dataset


def save_predicted_npz_group_by_series_id(
    predicted_npz_paths: Sequence[pathlib.Path], dataset_type: str, recreate: bool = False
):
    count_by_series_id_df = (
        child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df(
            dataset_type, as_polars=True
        )
        .group_by("series_id")
        .count()
        .collect()
    )
    min_duration_dict = dict(count_by_series_id_df.iter_rows())

    common_series_ids = None
    for predicted_npz_path in predicted_npz_paths:
        target_path = predicted_npz_path.with_stem(
            f"{predicted_npz_path.stem}-grouped-by-series_id"
        )
        if not target_path.exists():
            print(f"[Info] Create {target_path}")
        elif recreate and target_path.exists():
            print(f"[Info] Recreate {target_path}")
        else:
            continue

        data = np.load(predicted_npz_path)
        preds = data["pred"]
        series_ids = np.array([key.split("_")[0] for key in data["key"]])

        if common_series_ids is None:
            common_series_ids = series_ids
        else:
            assert np.all(common_series_ids == series_ids)

        np.savez(
            target_path,
            **{
                series_id: preds[series_ids == series_id].reshape(-1, 3)[
                    : min_duration_dict[series_id]
                ]
                for series_id in np.unique(series_ids)
            },
        )

    # first_series_ids, *other_series_ids = (pred.keys() for pred in preds_list)
    # for series_id in other_series_ids:
    #     assert first_series_ids == series_id
    #
    # min_duration_dict = [
    #     min(preds[series_id].shape[0] for preds in preds_list) for series_id in first_series_ids
    # ]
    # preds_list = [
    #     np.stack([preds[series_id][:min_duration] for preds in preds_list], axis=0)
    #     for series_id, min_duration in zip(first_series_ids, min_duration_dict)
    # ]
    # series_ids = np.array(
    #     [
    #         series_id
    #         for series_id, preds in zip(first_series_ids, preds_list, strict=True)
    #         for _ in range(preds.shape[1])
    #     ]
    # )
    # return series_ids, np.concatenate(preds_list, axis=1)


def load_predicted_npz_group_by_series_id(
    predicted_npz_paths: Sequence[pathlib.Path],
) -> tuple[tuple[str], list[NDArray[np.float_]]]:
    data_list = [
        np.load(predicted_npz_path.with_stem(f"{predicted_npz_path.stem}-grouped-by-series_id"))
        for predicted_npz_path in predicted_npz_paths
    ]
    common_series_ids = tuple(data_list[0].keys())

    assert common_series_ids is not None

    preds_list = [
        np.stack([data[series_id] for data in data_list], axis=0)
        for series_id in common_series_ids
    ]
    assert all(preds.shape[0] == len(predicted_npz_paths) for preds in preds_list)

    return common_series_ids, preds_list
