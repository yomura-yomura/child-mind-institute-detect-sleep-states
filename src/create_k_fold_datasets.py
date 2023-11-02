import argparse
import json
import pathlib

import numpy as np
import polars as pl
import sklearn.model_selection
import toml

import child_mind_institute_detect_sleep_states.data.comp_dataset

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=pathlib.Path)
args = parser.parse_args(
    [
        "config/multi_res_bi_gru.toml"
        #
    ]
)

with open(args.config_path) as f:
    config = toml.load(f)

data_dir_path = pathlib.Path("data")
data_dir_path /= "base"
# data_dir_path /= "with_part_id"
fold_type = "group"
# fold_type = "stratified_group"

df = pl.scan_parquet(data_dir_path / f"all-corrected-sigma{config['dataset']['sigma']}.parquet")
# n_total_records = df.select(pl.count()).collect()[0, 0]
# df = df.with_columns(index=pl.Series(np.arange(n_total_records, dtype=np.uint32)))

event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")

import pandas as pd


def get_cat_per_id(df: pd.DataFrame, target_cols: str = "series_id", nan_cols: str = "step") -> dict[str, float]:
    """get nanrate per user_id"""
    dict_result = {}
    for user_id, df in df.groupby(target_cols):
        nans = df[nan_cols].isna().sum()
        lens = df.shape[0]
        nan_rate = nans / lens
        dict_result[str(user_id)] = pd.cut(
            [nan_rate], bins=[-1] + [i / 10 for i in range(11)], labels=[i for i in range(11)]
        )[0]
    return dict_result


series_id = df.select("series_id").collect().to_numpy().astype(str).flatten()

if fold_type == "group":
    kf = sklearn.model_selection.GroupKFold(n_splits=config["train"]["n_folds"])
    kf_iter = kf.split(series_id, groups=series_id)
elif fold_type == "stratified_group":
    stratify_dict = get_cat_per_id(event_df)
    stratify = df.select(pl.col("series_id").map_dict(stratify_dict)).collect()
    kf = sklearn.model_selection.StratifiedGroupKFold(n_splits=config["train"]["n_folds"])
    kf_iter = kf.split(series_id, stratify, groups=series_id)
else:
    raise ValueError(f"unexpected {fold_type=}")


dst_data_dir_path = pathlib.Path("cmi-dss-train-k-fold-indices") / data_dir_path.relative_to("data")


for i_fold, (train_indices, valid_indices) in enumerate(kf_iter):
    # indices_dict = {"train": train_indices, "valid": valid_indices}
    # for dataset_type in ["train", "valid"]:
    #     p = (
    #         data_dir_path
    #         / fold_type
    #         / f"sigma{config['dataset']['sigma']}"
    #         / f"fold{i_fold}"
    #         / f"{dataset_type}.parquet"
    #     )
    #     if not p.exists():
    #         print(f"create {p}")
    #         p.parent.mkdir(parents=True, exist_ok=True)
    #         df.filter(pl.col("index").is_in(indices_dict[dataset_type])).drop("index").collect().write_parquet(p)
    p = dst_data_dir_path / fold_type / f"sigma{config['dataset']['sigma']}" / f"fold{i_fold}.npz"
    p.parent.mkdir(exist_ok=True, parents=True)
    print(p)
    np.savez(p, train=train_indices, valid=valid_indices)
    # with open(p, "w") as f:
    #     json.dump({"train": train_indices.tolist(), "valid": valid_indices.tolist()}, f)
