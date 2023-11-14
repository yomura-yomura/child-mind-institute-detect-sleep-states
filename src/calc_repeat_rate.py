import json
import pathlib

import numpy as np
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset

project_root_path = pathlib.Path(__file__).parent.parent


def get_repeat_rate(key_col: str = "anglez", shift_num: int = 17280) -> dict[str, float]:
    feat_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df("train")

    dict_result = {}

    for s_id, group_df in tqdm.tqdm(feat_df.groupby("series_id")):
        angle_df = group_df[[key_col]]
        rate_zero = sum(np.isclose(angle_df, angle_df.shift(shift_num)).astype(int))[0] / angle_df.shape[0]
        dict_result[str(s_id)] = rate_zero

    return dict_result


if __name__ == "__main__":
    rate_dict = get_repeat_rate()
    with open(project_root_path / "data" / "repeat_rate.json", "w") as f:
        json.dump(rate_dict, f)

    import pandas as pd

    df = pd.DataFrame.from_dict(rate_dict, orient="index", columns=["repeat_rate"])
    print(f"""{(df["repeat_rate"] > 0.6).value_counts() = }""")
