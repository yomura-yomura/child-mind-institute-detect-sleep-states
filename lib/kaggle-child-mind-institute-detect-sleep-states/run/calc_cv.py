import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset

predicted_fold_dir_path = pathlib.Path("tmp/predicted/ranchantan/exp041/train/fold_0/")

series_ids = [p.stem for p in predicted_fold_dir_path.glob("*.npz")]
event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
    "train"
).dropna()

sub_df_list = []
for series_id in tqdm.tqdm(series_ids):
    preds = np.load(predicted_fold_dir_path / f"{series_id}.npz")["arr_0"]
    sub_df_list.append(
        cmi_dss_lib.utils.post_process.post_process_for_seg(
            [series_id] * len(preds),
            preds,
            downsample_rate=2,
            score_th=0.005,
            distance=96,
            post_process_modes=None,
        )
    )
sub_df = pd.concat(sub_df_list)

score = cmi_dss_lib.utils.metrics.event_detection_ap(
    event_df[event_df["series_id"].isin(series_ids)], sub_df
)
print(f"{score = }")
