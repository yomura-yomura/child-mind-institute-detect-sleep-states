import pathlib
import pickle

import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde

import child_mind_institute_detect_sleep_states.data.comp_dataset

this_dir_path = pathlib.Path(__file__).parent


def get_kde() -> gaussian_kde:
    return gaussian_kde(np.load(this_dir_path / "start_timing_hour.npy"), bw_method=0.25)


if __name__ == "__main__":
    df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df(
        "train", as_polars=True
    )
    df = df.group_by("series_id").head(1).collect()

    data = df["timestamp"].str.to_datetime().dt.hour().to_numpy()
    np.save(this_dir_path / "start_timing_hour.npy", data)

    kde = get_kde()

    fig = px.histogram(x=data, histnorm="probability")
    x = np.arange(13, 24, 0.1)
    fig.add_trace({"mode": "lines", "x": x, "y": kde(x)})
    fig.show()

    fig = px.histogram(x=kde.resample(size=10_000)[0], histnorm="probability")
    fig.show()
