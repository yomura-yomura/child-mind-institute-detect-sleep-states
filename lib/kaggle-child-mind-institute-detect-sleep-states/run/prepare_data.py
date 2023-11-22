import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import sklearn.preprocessing
from cmi_dss_lib.utils.common import trace
from omegaconf import DictConfig
from scipy.signal import savgol_filter
from tqdm import tqdm

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829

STD_DICT = {"anglez": ANGLEZ_STD, "enmo": ENMO_STD}
MEAN_DICT = {"anglez": ANGLEZ_MEAN, "enmo": ENMO_MEAN}


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]

def count_unique(ser) -> int:
    return len(np.unique(ser))

def add_feature(series_df: pl.DataFrame, feature_names: list[str]) -> pl.DataFrame:
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
        *to_coord(pl.col("timestamp").dt.day(), 7, "week"),
    ).select("series_id", *feature_names)
    return series_df

def add_rolling_features(this_series_df: pl.DataFrame, rolling_features: list[str]) -> pl.DataFrame:
    """add rolling feature"""
    this_series_df = this_series_df.join(
        this_series_df.sort(by="index").set_sorted(column="index").rolling(index_column = "index",period="60i").agg(
            [
                pl.col("anglez_int").n_unique().cast(pl.Int16).alias("n_unique_anglez_5min"),
                pl.col("enmo_int").n_unique().cast(pl.Int16).alias("n_unique_enmo_5min"),
            ]
            ),
        on = "index")

    this_series_df = this_series_df.join(
        this_series_df.sort(by="index").set_sorted(column="index").rolling(index_column = "index",period="12i").agg(
            [
                pl.col("anglez_int").n_unique().cast(pl.Int16).alias("n_unique_anglez_1min"),
                pl.col("enmo_int").n_unique().cast(pl.Int16).alias("n_unique_enmo_1min"),
            ]
            ),
        on = "index")

    for col in ["anglez","enmo"]:
        # feature pct_change
        this_series_df = this_series_df.sort(by="index").with_columns((pl.col(col).diff()/pl.col(col).add(1e-6).shift(1)).abs().fill_null(0).alias(f"pct_change_{col}"))
        this_series_df = this_series_df.with_columns(pl.when(pl.col(f"pct_change_{col}") > 1.0).then(0).otherwise(pl.col(f"pct_change_{col}")).alias(f"pct_change_{col}"))

        for shift_min in [5,10,15,20,25,30]:
            this_series_df = this_series_df.with_columns(this_series_df[f"n_unique_{col}_5min"].shift(12*shift_min).fill_null(0).alias(f"n_unique_{col}_5min_{shift_min}minEarlier"))
        for shift_min in [3,6,9,12,15,18]:
            this_series_df = this_series_df.with_columns(this_series_df[f"n_unique_{col}_1min"].shift(12*shift_min).fill_null(0).alias(f"n_unique_{col}_1min_{shift_min}minEarlier"))

        this_series_df = this_series_df.with_columns(
                                    (
                                        pl.col(f"n_unique_{col}_5min_5minEarlier") +
                                        pl.col(f"n_unique_{col}_5min_10minEarlier") +
                                        pl.col(f"n_unique_{col}_5min_15minEarlier") +
                                        pl.col(f"n_unique_{col}_5min_20minEarlier") +
                                        pl.col(f"n_unique_{col}_5min_25minEarlier") +
                                        pl.col(f"n_unique_{col}_5min_30minEarlier") +
                                        pl.col(f"n_unique_{col}_5min")
                                    ).alias(f"rolling_unique_{col}_5min_sum"),
                                    (
                                        pl.col(f"n_unique_{col}_1min_3minEarlier") +
                                        pl.col(f"n_unique_{col}_1min_6minEarlier") +
                                        pl.col(f"n_unique_{col}_1min_9minEarlier") +
                                        pl.col(f"n_unique_{col}_1min_12minEarlier") +
                                        pl.col(f"n_unique_{col}_1min_15minEarlier") +
                                        pl.col(f"n_unique_{col}_1min_18minEarlier") +
                                        pl.col(f"n_unique_{col}_1min")
                                    ).alias(f"rolling_unique_{col}_1min_sum")
                                    )

    feature_savgol = ["anglez_lag_diff_abs", "enmo_lag_diff_abs"]
    ## savgol filter
    for feature in feature_savgol:
        this_series_df = this_series_df.with_columns(pl.Series(np.nan_to_num(savgol_filter(this_series_df[feature].clone().to_numpy(),720*5,3))).alias(feature + "_savgol"))
    
    scaler_featuers = rolling_features+feature_savgol
    # scalering
    scaler = sklearn.preprocessing.RobustScaler()
    this_series_df[scaler_featuers] = scaler.fit_transform(
        this_series_df[scaler_featuers].to_numpy()
    )

    this_series_df[scaler_featuers] = this_series_df[scaler_featuers].fill_nan(0)
    return this_series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path, save_as_npz: bool):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        if save_as_npz:
            np.savez_compressed(output_dir / f"{col_name}.npz", x)
        else:
            np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir = Path(cfg.dir.output_dir).resolve() / "prepare_data" / cfg.phase / cfg.scale_type
    print(f"{processed_dir = }")

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir, ignore_errors=True)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test", "dev"]:
            if cfg.phase in ["train", "dev"]:
                dataset_type = "train"
            else:
                dataset_type = "test"
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{dataset_type}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("anglez"),
                pl.col("enmo"),
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                # (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                # (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
                pl.col("anglez").cast(pl.Int16).alias("anglez_int"),
                pl.col("enmo").cast(pl.Int16).alias("enmo_int"),
                pl.col("anglez").diff(n=1).over("series_id").alias("anglez_lag_diff"),
                pl.col("enmo").diff(n=1).over("series_id").alias("enmo_lag_diff"),
                pl.col("anglez").diff(n=1).abs().over("series_id").alias("anglez_lag_diff_abs"),
                pl.col("enmo").diff(n=1).abs().over("series_id").alias("enmo_lag_diff_abs"),
                pl.col("anglez").diff(n=1).abs().cumsum().over("series_id").alias("anglez_lag_diff_abs_cumsum"),
                pl.col("enmo").diff(n=1).abs().cumsum().over("series_id").alias("enmo_lag_diff_abs_cumsum"),
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("anglez_int"),
                    pl.col("enmo_int"),
                    pl.col("anglez_lag_diff"),
                    pl.col("enmo_lag_diff"),
                    pl.col("anglez_lag_diff_abs"),
                    pl.col("enmo_lag_diff_abs"),
                    pl.col("timestamp"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )


        # add index col for rolling
        series_df = series_df.with_columns(series_df.select("series_id").with_row_count("index")["index"].cast(pl.Int32))
        #temp = series_df.rolling(index_column = "index",by="series_id",period="300i").agg([pl.col("anglez_int").n_unique().alias("n_unique_anglez_5min")])

        if cfg.scale_type == "constant":
            feature_names_to_preprocess = ["anglez", "enmo"]

            for feature_name in feature_names_to_preprocess:
                series_df[[feature_name]] = (
                    series_df[[feature_name]].to_numpy() - MEAN_DICT[feature_name]
                ) / STD_DICT[feature_name]
        elif cfg.scale_type == "robust_scaler":
            feature_names_to_preprocess = ["anglez", "enmo", "anglez_lag_diff", "enmo_lag_diff"]


            scaler = sklearn.preprocessing.RobustScaler()
            series_df[feature_names_to_preprocess] = scaler.fit_transform(
                series_df[feature_names_to_preprocess].to_numpy()
            )
        else:
            raise ValueError(f"unexpected {cfg.scale_type}")
        series_df[feature_names_to_preprocess] = series_df[feature_names_to_preprocess].fill_nan(0)

        n_unique = series_df.get_column("series_id").n_unique()

    feature_names = [
        *feature_names_to_preprocess,
        "anglez_int",
        "enmo_int",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
        "index", #for rolling features
        "anglez_lag_diff_abs",
        "enmo_lag_diff_abs",
        # "minute_sin",
        # "minute_cos",
    ]
    rolling_features = [
        "n_unique_anglez_5min",
        "n_unique_enmo_5min",
        "rolling_unique_anglez_5min_sum",
        "rolling_unique_enmo_5min_sum",
        "n_unique_anglez_1min",
        "n_unique_enmo_1min",
        "rolling_unique_anglez_1min_sum",
        "rolling_unique_enmo_1min_sum",
        "anglez_lag_diff_abs_savgol",
        "enmo_lag_diff_abs_savgol",
        ]
    pctchange_features = [
        "pct_change_anglez",
        "pct_change_enmo"
    ]

    print(f"{feature_names+rolling_features+pctchange_features = }")

    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
                # 特徴量を追加
                this_series_df = add_feature(this_series_df, feature_names)
                
                # NOTE: メモリーエラーを避けるためにここでrolling
                this_series_df = add_rolling_features(this_series_df,rolling_features)

                # 特徴量をそれぞれnpy/npzで保存
                
                
                series_dir = processed_dir / series_id
                save_each_series(this_series_df, feature_names+rolling_features+pctchange_features, series_dir, cfg.save_as_npz)


if __name__ == "__main__":
    main()
