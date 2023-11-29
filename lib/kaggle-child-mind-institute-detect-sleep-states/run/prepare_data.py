import pathlib
import pickle

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


def add_rolling_features(this_series_df: pl.DataFrame) -> pl.DataFrame:
    """add rolling feature"""

    # rolling n_unique features
    ## 5min features
    # this_series_df = this_series_df.join(
    #    this_series_df.sort(by="index").set_sorted(column="index").rolling(index_column = "index",period="60i",by = "series_id").agg(
    #        [
    #            pl.col("anglez_int").n_unique().cast(pl.Int16).alias("n_unique_anglez_5min"),
    #            pl.col("enmo_int").n_unique().cast(pl.Int16).alias("n_unique_enmo_5min"),
    #        ]
    #        ),
    #    on = "index")

    ## 1min features
    this_series_df = this_series_df.join(
        this_series_df.sort(by="index")
        .set_sorted(column="index")
        .rolling(index_column="index", period="12i", by="series_id")
        .agg(
            [
                pl.col("anglez_int").n_unique().cast(pl.Int16).alias("n_unique_anglez_1min"),
                pl.col("enmo_int").n_unique().cast(pl.Int16).alias("n_unique_enmo_1min"),
            ]
        ),
        on="index",
    )
    # rolling std features
    ## 5min features
    # this_series_df = this_series_df.join(
    #    this_series_df.sort(by="index").set_sorted(column="index").rolling(index_column = "index",period="60i",by = "series_id").agg(
    #        [
    #            pl.col("anglez").std().fill_null(0).alias("std_anglez_5min"),
    #            pl.col("enmo").std().fill_null(0).alias("std_enmo_5min"),
    #        ]
    #        ),
    #    on = "index")

    ## 1min features
    # this_series_df = this_series_df.join(
    #    this_series_df.sort(by="index").set_sorted(column="index").rolling(index_column = "index",period="12i",by = "series_id").agg(
    #        [
    #            pl.col("anglez").std().alias("std_anglez_1min"),
    #            pl.col("enmo").std().alias("std_enmo_1min"),
    #        ]
    #        ),
    #    on = "index")

    # for col in ["anglez","enmo"]:
    # shift rolling sum of n_uniques features
    # for shift_min in [5,10,15,20,25,30]:
    #    this_series_df = this_series_df.with_columns(this_series_df[f"n_unique_{col}_5min"].shift(12*shift_min).fill_null(0).alias(f"n_unique_{col}_5min_{shift_min}minEarlier"))
    # for shift_min in [3,6,9,12,15,18]:
    #    this_series_df = this_series_df.with_columns(this_series_df[f"n_unique_{col}_1min"].shift(12*shift_min).fill_null(0).alias(f"n_unique_{col}_1min_{shift_min}minEarlier"))

    # this_series_df = this_series_df.with_columns(
    #                            (
    #                                pl.col(f"n_unique_{col}_5min_5minEarlier") +
    #                                pl.col(f"n_unique_{col}_5min_10minEarlier") +
    #                                pl.col(f"n_unique_{col}_5min_15minEarlier") +
    #                                pl.col(f"n_unique_{col}_5min_20minEarlier") +
    #                                pl.col(f"n_unique_{col}_5min_25minEarlier") +
    #                                pl.col(f"n_unique_{col}_5min_30minEarlier") +
    #                                pl.col(f"n_unique_{col}_5min")
    #                            ).alias(f"rolling_unique_{col}_5min_sum"),
    #                            (
    #                                pl.col(f"n_unique_{col}_1min_3minEarlier") +
    #                                pl.col(f"n_unique_{col}_1min_6minEarlier") +
    #                                pl.col(f"n_unique_{col}_1min_9minEarlier") +
    #                                pl.col(f"n_unique_{col}_1min_12minEarlier") +
    #                                pl.col(f"n_unique_{col}_1min_15minEarlier") +
    #                                pl.col(f"n_unique_{col}_1min_18minEarlier") +
    #                                pl.col(f"n_unique_{col}_1min")
    #                            ).alias(f"rolling_unique_{col}_1min_sum")
    #                            )

    # feature_savgol = ["anglez_lag_diff_abs", "enmo_lag_diff_abs"]
    # savgol filter feature
    # for feature in feature_savgol:
    #    this_series_df = this_series_df.with_columns(pl.Series(np.nan_to_num(savgol_filter(this_series_df[feature].clone().to_numpy(),720*5,3))).alias(feature + "_savgol"))
    return this_series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: pathlib.Path, save_as_npz: bool):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        if save_as_npz:
            np.savez_compressed(output_dir / f"{col_name}.npz", x)
        else:
            np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    data_versions = cfg.data_version.split("+")
    processed_dir = pathlib.Path(cfg.dir.output_dir).resolve() / "prepare_data" / cfg.phase / cfg.scale_type
    print(f"{processed_dir = }")

    # # ディレクトリが存在する場合は削除
    # if processed_dir.exists():
    #     shutil.rmtree(processed_dir, ignore_errors=True)
    #     print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test", "dev"]:
            if cfg.phase in ["train", "dev"]:
                dataset_type = "train"
            else:
                dataset_type = "test"
            series_df = pl.scan_parquet(
                pathlib.Path(cfg.dir.data_dir) / f"{dataset_type}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        feature_names_to_preprocess_v1 = [
            "anglez",
            "enmo",
            "anglez_lag_diff",
            "enmo_lag_diff",
            "anglez_lag_diff_abs",
            "enmo_lag_diff_abs",
        ]
        feature_names_to_preprocess_v2 = [
            "anglez_lag_diff_abs_cumsum",
            "enmo_lag_diff_abs_cumsum",
            "pct_change_anglez",
            "pct_change_enmo",
            "rolling_std_1min_anglez",
            "rolling_std_1min_enmo",
        ]

        # preprocess
        series_df = series_df.with_columns(
            pl.col("anglez").cast(pl.Float32),
            pl.col("enmo").cast(pl.Float32),
            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
        )

        feature_names = []
        if "v1" in data_versions:
            feature_names += feature_names_to_preprocess_v1

        if "v1" in data_versions or "v2" in data_versions:
            series_df = series_df.with_columns(
                pl.col("anglez").diff(n=1).over("series_id").fill_null(0).cast(pl.Float32).alias("anglez_lag_diff"),
                pl.col("enmo").diff(n=1).over("series_id").fill_null(0).cast(pl.Float32).alias("enmo_lag_diff"),
            ).with_columns(
                pl.col("anglez_lag_diff").abs().cast(pl.Float32).alias("anglez_lag_diff_abs"),
                pl.col("enmo_lag_diff").abs().cast(pl.Float32).alias("enmo_lag_diff_abs"),
            )

        if "v2" in data_versions:
            series_df = (
                series_df.with_columns(
                    pl.col("anglez")
                    .rolling_std(window_size="12i", closed="both")
                    .over("series_id")
                    .cast(pl.Float32)
                    .alias("rolling_std_1min_anglez"),
                    pl.col("enmo")
                    .rolling_std(window_size="12i", closed="both")
                    .over("series_id")
                    .cast(pl.Float32)
                    .alias("rolling_std_1min_enmo"),
                )
                .with_columns(
                    (pl.col("anglez_lag_diff") / pl.col("anglez").add(1e-6))
                    .abs()
                    .over("series_id")
                    .cast(pl.Float32)
                    .alias("pct_change_anglez"),
                    (pl.col("enmo_lag_diff") / pl.col("enmo").add(1e-6))
                    .abs()
                    .over("series_id")
                    .cast(pl.Float32)
                    .alias("pct_change_enmo"),
                    pl.col("anglez_lag_diff_abs")
                    .cum_sum()
                    .over("series_id")
                    .cast(pl.Float32)
                    .alias("anglez_lag_diff_abs_cumsum"),
                    pl.col("enmo_lag_diff_abs")
                    .cum_sum()
                    .over("series_id")
                    .cast(pl.Float32)
                    .alias("enmo_lag_diff_abs_cumsum"),
                )
                .with_columns(
                    pl.when(pl.col(f"pct_change_anglez") > 1.0)
                    .then(0)
                    .otherwise(pl.col(f"pct_change_anglez"))
                    .alias(f"pct_change_anglez"),
                    pl.when(pl.col(f"pct_change_enmo") > 1.0)
                    .then(0)
                    .otherwise(pl.col(f"pct_change_enmo"))
                    .cast(pl.Float32)
                    .alias(f"pct_change_enmo"),
                )
            )
            feature_names += feature_names_to_preprocess_v2

        series_df = (
            series_df.select(
                [
                    pl.col("series_id"),
                    # pl.col("anglez"),
                    # pl.col("enmo"),
                    # pl.col("anglez_int"),
                    # pl.col("enmo_int"),
                    *map(pl.col, feature_names),
                    pl.col("timestamp"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
        # add index col for rolling
        # series_df = series_df.with_columns(series_df.select("series_id").with_row_count("index")["index"].cast(pl.Int32))

        if cfg.scale_type == "constant":
            feature_names_to_preprocess = ["anglez", "enmo"]

            for feature_name in feature_names_to_preprocess:
                series_df[[feature_name]] = (series_df[[feature_name]].to_numpy() - MEAN_DICT[feature_name]) / STD_DICT[
                    feature_name
                ]
        elif cfg.scale_type == "robust_scaler":
            preprocessing_scaler_dir = pathlib.Path(cfg.dir.preprocessing_scaler_dir)

            if "v1" in data_versions:
                features1 = series_df[feature_names_to_preprocess_v1].to_numpy()
                scaler_save_path1 = preprocessing_scaler_dir / "robust_scaler_v1.pkl"

                if cfg.just_load_scaler:
                    with open(scaler_save_path1, "rb") as f:
                        scaler1 = pickle.load(f)
                    print(f"[Info] RobustScaler has been loaded from {scaler_save_path1}")
                else:
                    scaler1 = sklearn.preprocessing.RobustScaler()
                    scaler1.fit(features1)
                    preprocessing_scaler_dir.mkdir(exist_ok=True)
                    with open(scaler_save_path1, "wb") as f:
                        pickle.dump(scaler1, f)
                    print(f"[Info] RobustScaler has been saved as {scaler_save_path1}")
                series_df[feature_names_to_preprocess_v1] = scaler1.transform(features1)

            if "v2" in data_versions:
                features2 = series_df[feature_names_to_preprocess_v2].to_numpy()
                scaler_save_path2 = preprocessing_scaler_dir / "robust_scaler_v2.pkl"

                if cfg.just_load_scaler:
                    with open(scaler_save_path2, "rb") as f:
                        scaler2 = pickle.load(f)
                    print(f"[Info] RobustScaler has been loaded from {scaler_save_path2}")
                else:
                    scaler2 = sklearn.preprocessing.RobustScaler()
                    scaler2.fit(features2)
                    preprocessing_scaler_dir.mkdir(exist_ok=True)
                    with open(scaler_save_path2, "wb") as f:
                        pickle.dump(scaler2, f)
                    print(f"[Info] RobustScaler has been saved as {scaler_save_path2}")
                series_df[feature_names_to_preprocess_v2] = scaler2.transform(features2)
        else:
            raise ValueError(f"unexpected {cfg.scale_type}")
        # series_df[feature_names_to_preprocess] = series_df[feature_names_to_preprocess].fill_nan(0)

    feature_names = [
        *feature_names,
        # "anglez_int",
        # "enmo_int",
        "hour_sin",
        "hour_cos",
        # "month_sin",
        # "month_cos",
        "week_sin",
        "week_cos",
        # "index", #for rolling features
        # "minute_sin",
        # "minute_cos",
    ]
    # add features
    rolling_features = [
        # "n_unique_anglez_5min",
        # "n_unique_enmo_5min",
        # "std_anglez_5min",
        # "std_enmo_5min",
        # "rolling_unique_anglez_5min_sum",
        # "rolling_unique_enmo_5min_sum",
        # "n_unique_anglez_1min",
        # "n_unique_enmo_1min",
        # "rolling_unique_anglez_1min_sum",
        # "rolling_unique_enmo_1min_sum",
        # "anglez_lag_diff_abs_savgol",
        # "enmo_lag_diff_abs_savgol",
    ]

    # print(f"{feature_names+rolling_features+pctchange_features = }")
    print(f"{feature_names+rolling_features = }")

    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df, feature_names)

            # NOTE: メモリーエラーを避けるためにここでrolling
            # if len(rolling_features) > 0:
            #    this_series_df = add_rolling_features(this_series_df)

            # 特徴量をそれぞれnpy/npzで保存

            series_dir = processed_dir / series_id
            save_each_series(this_series_df, feature_names + rolling_features, series_dir, cfg.save_as_npz)


if __name__ == "__main__":
    main()
