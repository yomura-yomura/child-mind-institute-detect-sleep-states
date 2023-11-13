import numpy as np
import pandas as pd
import polars as pl

import child_mind_institute_detect_sleep_states.data.comp_dataset


def get_repeat_rate(key_col: str = "anglez", shift_num: int = 17280) -> pd.DataFrame:
    feat_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df("train")

    dict_result = {}

    for s_id, group_df in feat_df.groupby("series_id"):
        angle_df = group_df[[key_col]]
        rate_zero = [sum(np.isclose((angle_df - angle_df.shift(shift_num)), 0.0).astype(int))[0] / angle_df.shape[0]]
        dict_result[s_id] = rate_zero

    return pd.DataFrame(dict_result)
