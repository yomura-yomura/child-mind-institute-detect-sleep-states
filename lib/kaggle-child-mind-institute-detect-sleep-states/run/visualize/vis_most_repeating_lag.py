import pathlib

import joblib
import numpy as np
import plotly.express as px
import tqdm

project_root_path = pathlib.Path(__file__).parent.parent.parent
data_dir_path = project_root_path / "output" / "prepare_data" / "train" / "robust_scaler"

series_ids = [p.name for p in data_dir_path.glob("*") if p.is_dir() and not p.name.startswith(".")]
lag_list = np.arange(int(0.8 * 24 * 60 * 60 // 5), int(1.2 * 24 * 60 * 60 // 5))


def get_counts(series_id: str):
    data = np.load(data_dir_path / series_id / "anglez.npy")
    return np.array(
        [np.count_nonzero(np.isclose(data[i:], data[: len(data) - i])) for i in lag_list]
    )


import cmi_dss_lib.utils.post_process

for series_id in series_ids:
    (
        start_indices_at_same,
        intervals,
    ) = cmi_dss_lib.utils.post_process.get_repeating_indices_and_intervals(
        data_dir_path, series_id
    )
    print(series_id, intervals)

fads

counts = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(get_counts)(series_id) for series_id in tqdm.tqdm(series_ids, position=0)
)
