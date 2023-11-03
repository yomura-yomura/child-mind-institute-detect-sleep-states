from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def split_unlabeled_intervals(
    df_event: pd.DataFrame, df_series: pd.DataFrame, add_interval: int = 12 * 60 * 12, min_col: int = 12 * 12 * 60 * 2
):
    """
    df_event and df_series is  groupby(series_id)
    add_interval = 12h
    """
    max_step = df_series["step"].max()
    if df_event["step"].isna().sum() == 0:
        return []
    list_del = find_del_intervals(df=df_event, max_step=max_step)
    list_idx = []
    last_step = len(list_del) - 1
    for step, (start, end) in enumerate(list_del):
        mask1 = df_series["step"] < start + add_interval
        mask2 = df_series["step"] > end - add_interval
        mask_idx1 = df_series[mask1].index
        mask_idx2 = df_series[mask2].index
        list_idx.append(list(mask_idx1))
        df_series = df_series.loc[mask2]
        if (end != max_step) and (df_series.shape[0] > min_col) and (step == last_step):
            list_idx.append(list(mask_idx2))

    return list_idx


def find_del_intervals(
    df: pd.DataFrame, max_step: int, col_night: str = "night", col_step: str = "step"
) -> list[tuple[int, int]]:
    """find missing intervals step
    df: event
    """
    list_del_intervals = []
    last_night = df[col_night].max()
    first_night = df[col_night].min()
    for i_night in df[col_night].unique():
        if df[df[col_night] == i_night][col_step].isna().sum() > 0:
            i_m1_night = i_night - 1
            i_p1_night = i_night + 1
            if i_night == first_night:
                last_step = df[df[col_night] == i_p1_night][col_step].values[0]
                list_del_intervals.append((0, last_step))
            elif i_night == last_night:
                first_step = df[df[col_night] == i_m1_night][col_step].values[1]
                list_del_intervals.append((first_step, max_step))
            else:
                first_step = df[df[col_night] == i_m1_night][col_step].values[1]
                last_step = df[df[col_night] == i_p1_night][col_step].values[0]
                list_del_intervals.append((first_step, last_step))
    return process_list(list_del_intervals)


def process_list(input_list):
    output_list = []
    for i in range(len(input_list)):
        if not np.isnan(input_list[i][0]):
            for j in range(i, len(input_list)):
                if not np.isnan(input_list[j][1]):
                    output_list.append((input_list[i][0], input_list[j][1]))
                    break
    return output_list


if __name__ == "__main__":
    path = Path("data/child-mind-institute-detect-sleep-states")
    train = pd.read_parquet(path / "train_series.parquet")
    event = pd.read_csv(path / "train_events.csv")

    train["part_id"] = np.nan
    list_id = train["series_id"].unique()
    for user_id in tqdm(list_id):
        list_df = split_unlabeled_intervals(
            df_event=event[event["series_id"] == user_id], df_series=train[train["series_id"] == user_id]
        )
        if len(list_df) > 0:
            for num, list_idx in enumerate(list_df):
                assert int(train.loc[list_idx, "part_id"].isna().sum()) == int(train.loc[list_idx, "part_id"].shape[0])
                train.loc[list_idx, "part_id"] = train.loc[list_idx, "series_id"] + f"_{num}"
        else:
            train.loc[train["series_id"] == user_id, "part_id"] = user_id
    train.to_parquet(path / "train_part_series.parquet")
