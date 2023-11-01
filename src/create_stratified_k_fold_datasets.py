import argparse
import pathlib

import numpy as np
import polars as pl
import sklearn.model_selection
import toml

import child_mind_institute_detect_sleep_states.data.comp_dataset
from child_mind_institute_detect_sleep_states.common.get_nan_rate import get_cat_per_id

parser = argparse.ArgumentParser()
parser.add_argument("config_path",default="", type=pathlib.Path)
args = parser.parse_args()

with open(args.config_path) as f:
    config = toml.load(f)

data_dir_path = pathlib.Path("data")


df = pl.scan_parquet(data_dir_path / f"all-corrected-sigma{config['dataset']['sigma']}.parquet")
n_total_records = df.select(pl.count()).collect()[0, 0]
df = df.with_columns(index=pl.Series(np.arange(n_total_records, dtype=np.uint32)))

event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")

dict_id2cat = get_cat_per_id(df = event_df)
series_id = df.select("series_id").collect().to_numpy().astype(str).flatten()
nan_label = [dict_id2cat[user_id] for user_id in series_id]
kf = sklearn.model_selection.StratifiedGroupKFold(n_splits=config["train"]["n_folds"])
for i_fold, (train_indices, valid_indices) in enumerate(kf.split(X=series_id,y=nan_label,groups=series_id)):
    p = data_dir_path / f"sigma{config['dataset']['sigma']}_stratified" / f"fold{i_fold}" / "train.parquet"
    if not p.exists():
        print(f"create {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("index").is_in(train_indices)).drop("index").collect().write_parquet(p)

    p = data_dir_path / f"sigma{config['dataset']['sigma']}_stratified" / f"fold{i_fold}" / "valid.parquet"
    if not p.exists():
        print(f"create {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("index").is_in(valid_indices)).drop("index").collect().write_parquet(p)
    # break
