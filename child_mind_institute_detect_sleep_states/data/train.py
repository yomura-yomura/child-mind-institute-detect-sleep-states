import polars as pl

from .. import pj_struct_paths


def get_train_df(sigma: int, train_dataset_type: str) -> pl.LazyFrame:
    df = pl.scan_parquet(
        pj_struct_paths.get_data_dir_path() / "cmi-dss-train-datasets" / "base" / f"all-corrected-sigma{sigma}.parquet"
    )

    if train_dataset_type == "with_part_id":
        part_id_df = pl.scan_parquet(
            pj_struct_paths.get_data_dir_path() / "train-series-with-partid" / "train_series.parquet"
        ).select(pl.col(["series_id", "step", "part_id"]))
        df = df.join(part_id_df, on=["series_id", "step"], how="left")
    return df
