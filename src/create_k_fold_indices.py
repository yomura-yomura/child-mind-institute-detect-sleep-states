import argparse
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import sklearn.model_selection
import toml

import child_mind_institute_detect_sleep_states.data.comp_dataset

project_root_path = pathlib.Path(__file__).parent
train_dataset_dir_path = project_root_path / "data" / "cmi-dss-train-datasets"


train_dataset_type = "base"
# train_dataset_type = "with_part_id"

# fold_type = "group"
fold_type = "stratified_group"


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


df = pl.scan_parquet(train_dataset_dir_path / train_dataset_type / f"all.parquet")

event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")


def get_cat_per_id(df_: pd.DataFrame, target_cols: str = "series_id", nan_cols: str = "step") -> dict[str, float]:
    """get nan-rate per user_id"""
    dict_result = {}
    for user_id, df_ in df_.groupby(target_cols):
        nans = df_[nan_cols].isna().sum()
        lens = df_.shape[0]
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


dst_fold_dir_path = project_root_path / "data" / "cmi-dss-train-k-fold-indices" / train_dataset_type / fold_type


for i_fold, (train_indices, valid_indices) in enumerate(kf_iter):
    p = dst_fold_dir_path / f"fold{i_fold}.npz"
    p.parent.mkdir(exist_ok=True, parents=True)
    print(p)

    train_sel = np.zeros(len(series_id), dtype=bool)
    train_sel[train_indices] = True
    valid_sel = np.zeros(len(series_id), dtype=bool)
    valid_sel[valid_indices] = True
    np.savez_compressed(p, train=train_sel, valid=valid_sel)
    # with open(p, "w") as f:
    #     json.dump({"train": train_indices.tolist(), "valid": valid_indices.tolist()}, f)
