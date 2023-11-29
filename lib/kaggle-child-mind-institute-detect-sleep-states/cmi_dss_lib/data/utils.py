from typing import Literal, cast

import pandas as pd
import polars as pl

import child_mind_institute_detect_sleep_states.data.comp_dataset


def get_duration_dict(phase: Literal["train", "test", "dev"]) -> dict[str, int]:
    dataset_type = cast(Literal["train", "test"], "test" if phase == "test" else "train")
    count_by_series_id_df = (
        child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df(
            dataset_type, as_polars=True
        )
        .group_by("series_id")
        .count()
        .collect()
    )
    return dict(count_by_series_id_df.iter_rows())


def get_start_timing_dict(phase: Literal["train", "test", "dev"]) -> dict[str, pd.Timestamp]:
    dataset_type = cast(Literal["train", "test"], "test" if phase == "test" else "train")
    timestamp_by_series_id_df = (
        child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df(
            dataset_type, as_polars=True
        )
        .sort(["series_id", "step"])
        .group_by("series_id")
        .head(1)
        .select(["series_id", "timestamp"])
        .with_columns(pl.col("timestamp").str.to_datetime())
        .collect()
    )
    return dict(timestamp_by_series_id_df.to_numpy())
