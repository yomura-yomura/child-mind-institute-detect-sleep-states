import pathlib

import polars as pl

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.pj_struct_paths
from child_mind_institute_detect_sleep_states.data.comp_dataset import event_mapping

project_root_path = pathlib.Path(__file__).parent
train_dataset_dir_path = project_root_path / "data" / "cmi-dss-train-datasets"


# train_dataset_type = "base"
train_dataset_type = "with_part_id"

data_dir_path = train_dataset_dir_path / train_dataset_type
data_dir_path.mkdir(exist_ok=True, parents=True)

child_mind_institute_detect_sleep_states.pj_struct_paths.set_pj_struct_paths(
    kaggle_dataset_dir_path=child_mind_institute_detect_sleep_states.pj_struct_paths.get_data_dir_path()
    / "train-series-with-partid"
)

df_dict = child_mind_institute_detect_sleep_states.data.comp_dataset.get_df_dict("train", as_polars=True)

# if train_dataset_type == "with_part_id":
#     target_unique_column = "part_id"
# else:
#     target_unique_column = "part_id"
#     df_dict["series"] = (
#         df_dict["series"].rename({"part_id": "series_id"})
#     )


all_dataset_path = data_dir_path / "all.parquet"

if all_dataset_path.exists():
    print(f"load {all_dataset_path}")
    df = pl.read_parquet(all_dataset_path)
else:
    print(f"create {all_dataset_path}")
    df = (
        df_dict["series"]
        .with_columns(
            [
                pl.col("step").cast(pl.UInt32),
                pl.col("timestamp").str.to_datetime(time_unit="ms"),
                pl.col("anglez").cast(pl.Float32),
                pl.col("enmo").cast(pl.Float32),
            ]
        )
        .join(
            df_dict["event"].with_columns(
                [
                    pl.col("step").cast(pl.UInt32),
                    pl.col("timestamp").str.to_datetime(time_unit="ms"),
                    pl.col("event").map_dict(event_mapping).cast(pl.UInt8),
                    pl.col("night").cast(pl.UInt32),
                ]
            ),
            on=["series_id", "step", "timestamp"],
            how="left",
        )
        .filter(pl.col("step").is_not_nan())
        .with_columns([pl.col("event").fill_null(0)])
        .sort(by=["series_id", "step"], descending=False)
        .collect()
    )
    df.write_parquet(all_dataset_path)


df = df.to_pandas()

import numpy as np
import pandas as pd
import scipy.stats

for label, i in event_mapping.items():
    df[f"event_{label}"] = pd.get_dummies(df["event"])[i]
df = df.drop(columns=["event"])

# sigma = 1
# sigma = 12
# sigma = 12 * 60
sigma = 12 * 9

steps = np.arange(-sigma * 10, sigma * 10 + 1)
norm_probs = scipy.stats.norm.pdf(steps, loc=0, scale=sigma)

sel = norm_probs > 1e-5
norm_probs = norm_probs[sel]
steps = steps[sel]

# normalize to 1
# norm_probs /= norm_probs.max()

sigma_in_min = len(norm_probs) * 12 / 60
print(f"{sigma_in_min = :.1f} min")


df_ = pd.merge(
    *[
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "step": np.ravel(df[df[key]]["step"].to_numpy()[:, np.newaxis] + steps),
                        key: np.ravel(np.repeat([norm_probs], len(df[df[key]]), axis=0)),
                    }
                )
                for series_id, df in df[df["event_onset"] | df["event_wakeup"]].groupby("series_id")
            ]
        )
        for key in ["event_onset", "event_wakeup"]
    ],
    on=["series_id", "step"],
    how="outer",
)

df_ = df[["series_id", "step", "timestamp", "anglez", "enmo", "night", "event_others"]].merge(
    df_, on=["series_id", "step"], how="left"
)
df = df_[["series_id", "step", "timestamp", "anglez", "enmo", "night", "event_others", "event_onset", "event_wakeup"]]
df["event_others"] = df["event_others"].astype("f8")

cols = [f"event_{k}" for k in event_mapping]
for col in cols:
    df[col] = df[col].fillna(0)


# normalize
# summed = df[cols].sum(axis=1)
# df[cols] /= summed.to_numpy()[:, np.newaxis]

df.to_parquet(data_dir_path / f"all-corrected-sigma{sigma}.parquet")
