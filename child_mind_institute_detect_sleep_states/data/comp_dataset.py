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


def get_submission_df(preds, series_id: str | Sequence[str], calc_type: Literal["max-along-type", "top-probs"]):
    if calc_type == "max-along-type":
        indices = np.argmax(preds, axis=1)

        if isinstance(series_id, str):
            pass
        else:
            series_id = series_id[indices > 0]

        submission_df = pd.DataFrame(
            {
                "series_id": series_id,
                "step": np.where(indices > 0)[0],
                "event": indices[indices > 0],
                "score": np.max(preds[indices > 0], axis=1),
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
        n_days = max(1, round(len(preds) * 5 / (24 * 60 * 60)))
        onset_indices = np.argsort(preds[:, 1])[-n_days:]
        wakeup_indices = np.argsort(preds[:, 2])[-n_days:]

        submission_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "series_id": series_id[onset_indices],
                        "step": onset_indices + 1,
                        "event": "onset",
                        "score": preds[onset_indices, 1],
                    }
                ),
                pd.DataFrame(
                    {
                        "series_id": series_id[wakeup_indices],
                        "step": wakeup_indices + 1,
                        "event": "wakeup",
                        "score": preds[wakeup_indices, 2],
                    }
                ),
            ]
        )
        submission_df = submission_df.sort_values(["series_id", "step"]).reset_index(drop=True)
    return submission_df
