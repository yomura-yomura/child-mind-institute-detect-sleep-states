import argparse
import pathlib

import numpy as np
import polars as pl
import sklearn.model_selection
import toml

import child_mind_institute_detect_sleep_states.data.comp_dataset

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=pathlib.Path)
args = parser.parse_args()

with open(args.config_path) as f:
    config = toml.load(f)

data_dir_path = pathlib.Path("data")


df = pl.scan_parquet(data_dir_path / f"all-corrected-sigma{config['dataset']['sigma']}.parquet")
n_total_records = df.select(pl.count()).collect()[0, 0]
df = df.with_columns(index=pl.Series(np.arange(n_total_records, dtype=np.uint32)))

event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")


series_id = df.select("series_id").collect().to_numpy().astype(str).flatten()
kf = sklearn.model_selection.GroupKFold(n_splits=config["train"]["n_folds"])
for i_fold, (train_indices, valid_indices) in enumerate(kf.split(series_id, groups=series_id)):
    p = data_dir_path / f"sigma{config['dataset']['sigma']}" / f"fold{i_fold}" / "train.parquet"
    if not p.exists():
        print(f"create {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("index").is_in(train_indices)).drop("index").collect().write_parquet(p)

    p = data_dir_path / f"sigma{config['dataset']['sigma']}" / f"fold{i_fold}" / "valid.parquet"
    if not p.exists():
        print(f"create {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("index").is_in(valid_indices)).drop("index").collect().write_parquet(p)
    # break
