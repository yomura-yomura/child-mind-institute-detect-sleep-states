import json
import pathlib
import pickle

import numpy as np
import plotly.express as px
import polars as pl
from scipy.stats import gaussian_kde

import child_mind_institute_detect_sleep_states.data.comp_dataset

this_dir_path = pathlib.Path(__file__).parent


def get_start_timing_hour_dict() -> dict:
    with open(this_dir_path / "start_timing_hour.json", "r") as f:
        return json.load(f)


def get_kde() -> gaussian_kde:
    return gaussian_kde(list(get_start_timing_hour_dict().values()), bw_method=0.15)


if __name__ == "__main__":
    df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df("train", as_polars=True)
    df = (
        df.group_by("series_id")
        .head(1)
        .with_columns(pl.col("timestamp").str.to_datetime())
        .with_columns(start_timing_hour=pl.col("timestamp").dt.hour() + pl.col("timestamp").dt.minute() / 60)
        .collect()
    )
    assert all(df["timestamp"].dt.minute().unique() == [0, 15, 30, 45])
    data = df["start_timing_hour"].to_numpy()
    with open(this_dir_path / "start_timing_hour.json", "w") as f:
        json.dump(dict(df[["series_id", "start_timing_hour"]].iter_rows()), f)

    kde = get_kde()

    fig = px.histogram(x=data, histnorm="probability density", barmode="group", nbins=25).update_traces(
        name="data", showlegend=True
    )
    x = np.arange(13, 24, 0.1)
    fig.add_trace(dict(name="kde", mode="lines", x=x, y=kde(x)))

    fig.add_trace(
        px.histogram(x=kde.resample(size=10_000)[0], histnorm="probability density")
        .update_traces(marker_color="green", name="resampled", showlegend=True)
        .data[0]
    )
    fig.show()
