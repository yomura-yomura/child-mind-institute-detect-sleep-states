import math
import os
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


# should be moved to proper place
import itertools
from typing import Iterable

from numpy.typing import NDArray


def get_predicted_group_by_series_id(data_list: Iterable[NDArray[np.float_]]):
    preds_list = [
        {
            series_id: np.concatenate([preds for _, preds in g])
            for series_id, g in itertools.groupby(
                zip([key.split("_")[0] for key in data["key"]], data["pred"]),
                key=lambda p: p[0],
            )
        }
        for data in data_list
    ]
    first_key, *other_keys = (pred.keys() for pred in preds_list)
    for key in other_keys:
        assert first_key == key

    min_durations = [
        min(preds[series_id].shape[0] for preds in preds_list) for series_id in first_key
    ]
    preds_list = [
        np.stack([preds[series_id][:min_duration] for preds in preds_list], axis=0)
        for series_id, min_duration in zip(first_key, min_durations)
    ]
    keys = np.array(
        [
            key
            for key, preds in zip(first_key, preds_list, strict=True)
            for _ in range(preds.shape[1])
        ]
    )
    return keys, np.concatenate(preds_list, axis=1)
