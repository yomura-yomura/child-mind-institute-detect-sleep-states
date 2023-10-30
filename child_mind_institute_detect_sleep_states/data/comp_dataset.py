import pathlib
from typing import Literal, Sequence, TypeAlias, overload

import numpy as np
import pandas as pd
import polars as pl

project_root_path = pathlib.Path(__file__).parent.parent.parent
data_dir_path = project_root_path / "data"

DatasetType: TypeAlias = Literal["train", "test"]

event_mapping = {"others": 0, "onset": 1, "wakeup": 2}


@overload
def get_df_dict(dataset_type: DatasetType, as_polars: Literal[False] = False) -> dict[str, pd.DataFrame]:
    ...


@overload
def get_df_dict(dataset_type: DatasetType, as_polars: Literal[True]) -> dict[str, pl.LazyFrame]:
    ...


def get_df_dict(dataset_type: DatasetType, as_polars: bool = False) -> dict[str, pd.DataFrame | pl.LazyFrame]:
    return {"event": get_event_df(dataset_type, as_polars), "series": get_series_df(dataset_type, as_polars)}


def get_event_df(dataset_type: DatasetType, as_polars: bool = False) -> pd.DataFrame | pl.LazyFrame:
    path = data_dir_path / "child-mind-institute-detect-sleep-states" / f"{dataset_type}_events.csv"
    if as_polars:
        return pl.scan_csv(path)
    else:
        return pd.read_csv(path)


def get_series_df(dataset_type: DatasetType, as_polars: bool = False) -> pd.DataFrame | pl.LazyFrame:
    path = data_dir_path / "child-mind-institute-detect-sleep-states" / f"{dataset_type}_series.parquet"
    if as_polars:
        return pl.scan_parquet(path)
    else:
        return pd.read_parquet(path)


def get_submission_df(preds, series_id: str | Sequence[str], steps, calc_type: Literal["max-along-type", "top-probs"]):
    if calc_type == "max-along-type":
        assert False
        preds = np.concatenate(preds, axis=0)
        indices = np.argmax(preds, axis=1)

        if isinstance(series_id, str):
            pass
        else:
            series_id = np.repeat(np.expand_dims(series_id, axis=1), indices.shape[1], axis=1)[indices > 0].flatten()

        submission_df = pd.DataFrame(
            {
                "series_id": series_id,
                "step": steps[indices > 0].flatten(),
                "event": indices[indices > 0],
                "score": np.max(preds, axis=2)[indices > 0],
            }
        )
        submission_df["event"] = submission_df["event"].map({v: k for k, v in event_mapping.items()})

        import itertools

        event_iter = itertools.cycle(["onset", "wakeup"])
        target_event = next(event_iter)
        cnt = 1

        idx_list = []
        for idx, (cur_event, next_event) in enumerate(zip(submission_df["event"], submission_df["event"].shift(-1))):
            if target_event != cur_event:
                idx_list.append(np.nan)
            else:
                idx_list.append(cnt)
                if next_event is not None and cur_event == next_event:
                    pass
                else:
                    target_event = next(event_iter)
                    cnt += 1

            if next_event is None:
                break

        submission_df["night"] = idx_list

        submission_df = submission_df.dropna()
        submission_df["night"] = submission_df["night"].astype("i8")
        submission_df = (
            submission_df.sort_values(["night", "score"], ascending=[True, False]).groupby(["event", "night"]).head(1)
        )
    elif calc_type == "top-probs":
        n_days = max(1, round(steps.max() * 5 / (24 * 60 * 60)))

        def rolling(a, window, axis, writable: bool = False):
            shape = a.shape[:axis] + (a.shape[axis] - window + 1, window) + a.shape[axis + 1 :]
            strides = a.strides[:axis] + (a.strides[axis],) + a.strides[axis:]
            rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=writable)
            return rolling

        window = 40 + 1

        assert preds.shape[-1] >= 2
        interest = slice((window - 1) // 2, -(window - 1) // 2)
        corrected_preds_list = []
        for i in range(2):
            max_preds = np.max(rolling(preds[..., -i], window=window, axis=1), axis=2)
            max_preds[preds[:, interest, -i] != max_preds] = 0
            corrected_preds = np.zeros(preds.shape[:-1], dtype=preds.dtype)
            corrected_preds[:, interest] = max_preds
            corrected_preds_list.append(corrected_preds)

        onset_indices = np.argsort(corrected_preds_list[0], axis=1)[:, -n_days:]
        wakeup_indices = np.argsort(corrected_preds_list[1], axis=1)[:, -n_days:]
        # print(corrected_preds_list[0][0][wakeup_indices[0]])

        series_id = np.repeat(np.expand_dims(series_id, axis=1), preds.shape[1], axis=1)

        submission_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "series_id": np.take_along_axis(series_id, onset_indices, axis=1).flatten(),
                        "step": np.take_along_axis(steps, onset_indices, axis=1).flatten(),
                        "event": "onset",
                        "score": np.take_along_axis(preds[..., -2], onset_indices, axis=1).flatten(),
                    }
                ),
                pd.DataFrame(
                    {
                        "series_id": np.take_along_axis(series_id, wakeup_indices, axis=1).flatten(),
                        "step": np.take_along_axis(steps, wakeup_indices, axis=1).flatten(),
                        "event": "wakeup",
                        "score": np.take_along_axis(preds[..., -1], wakeup_indices, axis=1).flatten(),
                    }
                ),
            ]
        )
        submission_df = submission_df.sort_values(["series_id", "step"]).reset_index(drop=True)
    else:
        raise ValueError(f"unexpected {calc_type=}")

    submission_df = (
        submission_df.sort_values(["series_id", "step", "event", "score"], ascending=[True, True, True, False])
        .groupby(["series_id", "step", "event"])
        .head(1)
    )

    return submission_df
