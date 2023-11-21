import pathlib
import pickle
import shutil

import hydra
import numpy as np
import polars as pl
import sklearn.preprocessing
from cmi_dss_lib.config import PrepareDataConfig
from cmi_dss_lib.utils.common import trace
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


def add_rolling_features(
    this_series_df: pl.DataFrame, rolling_features: list[str]
) -> pl.DataFrame:
    """add rolling feature"""
    this_series_df = this_series_df.join(
        this_series_df.sort(by="index")
        .set_sorted(column="index")
        .rolling(index_column="index", period="60i")
        .agg(
            [
                pl.col("anglez_int").n_unique().cast(pl.Int16).alias("n_unique_anglez_5min"),
                pl.col("enmo_int").n_unique().cast(pl.Int16).alias("n_unique_enmo_5min"),
            ]
        ),
        on="index",
    )

    for col in ["anglez", "enmo"]:
        for shift_min in [5, 10, 15, 20, 25, 30]:
            this_series_df = this_series_df.with_columns(
                this_series_df[f"n_unique_{col}_5min"]
                .shift(12 * shift_min)
                .fill_null(0)
                .alias(f"n_unique_{col}_5min_{shift_min}minEarlier")
            )

        this_series_df = this_series_df.with_columns(
            (
                pl.col(f"n_unique_{col}_5min_5minEarlier")
                + pl.col(f"n_unique_{col}_5min_10minEarlier")
                + pl.col(f"n_unique_{col}_5min_15minEarlier")
                + pl.col(f"n_unique_{col}_5min_20minEarlier")
                + pl.col(f"n_unique_{col}_5min_25minEarlier")
                + pl.col(f"n_unique_{col}_5min_30minEarlier")
                + pl.col(f"n_unique_{col}_5min")
            ).alias(f"rolling_unique_{col}_sum")
        )

    # scalering
    scaler = sklearn.preprocessing.RobustScaler()
    this_series_df[rolling_features] = scaler.fit_transform(
        this_series_df[rolling_features].to_numpy()
    )

    return this_series_df


def save_each_series(
    this_series_df: pl.DataFrame, columns: list[str], output_dir: pathlib.Path, save_as_npz: bool
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        if save_as_npz:
            np.savez_compressed(output_dir / f"{col_name}.npz", x)
        else:
            np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir = (
        pathlib.Path(cfg.dir.output_dir).resolve() / "prepare_data" / cfg.phase / cfg.scale_type
    )
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
                pathlib.Path(cfg.dir.data_dir) / f"{dataset_type}_series.parquet",
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
        series_df = series_df.with_columns(
            series_df.select("series_id").with_row_count("index")["index"].cast(pl.Int32)
        )
        # temp = series_df.rolling(index_column = "index",by="series_id",period="300i").agg([pl.col("anglez_int").n_unique().alias("n_unique_anglez_5min")])

        if cfg.scale_type == "constant":
            feature_names_to_preprocess = ["anglez", "enmo"]

            for feature_name in feature_names_to_preprocess:
                series_df[[feature_name]] = (
                    series_df[[feature_name]].to_numpy() - MEAN_DICT[feature_name]
                ) / STD_DICT[feature_name]
        elif cfg.scale_type == "robust_scaler":
            feature_names_to_preprocess = [
                "anglez",
                "enmo",
                "anglez_lag_diff",
                "enmo_lag_diff",
                "anglez_lag_diff_abs",
                "enmo_lag_diff_abs",
            ]
            features = series_df[feature_names_to_preprocess].to_numpy()

            preprocessing_scaler_dir = pathlib.Path(cfg.dir.preprocessing_scaler_dir)
            scaler_save_path = preprocessing_scaler_dir / "robust_scaler.pkl"
            if cfg.just_load_scaler:
                with open(scaler_save_path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"[Info] RobustScaler has been loaded from {scaler_save_path}")
            else:
                scaler = sklearn.preprocessing.RobustScaler()
                scaler.fit(features)
                preprocessing_scaler_dir.mkdir(exist_ok=True)
                with open(scaler_save_path, "wb") as f:
                    pickle.dump(scaler, f)
                print(f"[Info] RobustScaler has been saved as {scaler_save_path}")
            series_df[feature_names_to_preprocess] = scaler.transform(features)
        else:
            raise ValueError(f"unexpected {cfg.scale_type}")
        series_df[feature_names_to_preprocess] = series_df[feature_names_to_preprocess].fill_nan(0)

        n_unique = series_df.get_column("series_id").n_unique()

    feature_names = [
        *feature_names_to_preprocess,
        "hour_sin",
        "hour_cos",
        # "month_sin",
        # "month_cos",
        "week_sin",
        "week_cos",
        # "minute_sin",
        # "minute_cos",
        # "anglez_int",
        # "enmo_int",
    ]
    rolling_features = cfg.rolling_features or []
    # rolling_features = [
    #     "n_unique_anglez_5min",
    #     "n_unique_enmo_5min",
    #     "rolling_unique_anglez_sum",
    #     "rolling_unique_enmo_sum",
    # ]
    if len(rolling_features) > 0:
        feature_names.append("index")
        if "anglez_int" not in feature_names:
            feature_names.append("anglez_int")
        if "enmo_int" not in feature_names:
            feature_names.append("enmo_int")

    print(f"{feature_names+rolling_features = }")

    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df, feature_names)

            # NOTE: メモリーエラーを避けるためにここでrolling
            if len(rolling_features) > 0:
                this_series_df = add_rolling_features(this_series_df, rolling_features)

            # 特徴量をそれぞれnpy/npzで保存

            series_dir = processed_dir / series_id
            save_each_series(
                this_series_df, feature_names + rolling_features, series_dir, cfg.save_as_npz
            )


if __name__ == "__main__":
    main()
