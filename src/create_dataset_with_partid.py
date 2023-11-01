from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from child_mind_institute_detect_sleep_states.common.split_unlabeled_intervals import split_unlabeled_intervals

path = Path("data/child-mind-institute-detect-sleep-states")
train = pd.read_parquet(path /"train_series.parquet")
event = pd.read_csv(path /"train_events.csv")


train["part_id"] = np.nan
list_id = train["series_id"].unique()
for user_id in tqdm(list_id):
    list_df = split_unlabeled_intervals(df_event = event[event["series_id"]==user_id],df_series=train[train["series_id"]==user_id])
    if len(list_df) > 0:
        for num,list_idx in enumerate(list_df):
            assert int(train.loc[list_idx,"part_id"].isna().sum()) == int(train.loc[list_idx,"part_id"].shape[0])
            train.loc[list_idx,"part_id"] = train.loc[list_idx,"series_id"] + f"_{num}"
    else:
        train.loc[train["series_id"]==user_id,"part_id"] = user_id
train.to_parquet(path / "train_part_series.parquet")