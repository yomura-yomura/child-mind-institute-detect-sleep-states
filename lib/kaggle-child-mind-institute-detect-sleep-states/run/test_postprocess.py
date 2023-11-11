import itertools
import pathlib
from typing import Literal

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import matplotlib.pyplot as plt
import numpy as np
import numpy_utility as npu
import tqdm
from cmi_dss_lib.utils.post_process import PostProcessModes

import child_mind_institute_detect_sleep_states.data.comp_dataset
from child_mind_institute_detect_sleep_states.data.comp_dataset import rolling

project_root_path = pathlib.Path(__file__).parent.parent


post_process_modes = {
    "sleeping_edges_as_probs": cmi_dss_lib.utils.post_process.SleepingEdgesAsProbsSetting(
        sleep_prob_th=0.2, min_sleeping_hours=6
    ),
    "cutting_probs_by_sleep_prob": cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
        watch_interval_hour=6, sleep_occupancy_th=0.3
    ),
}


def get_corrected_pred(all_data, time_window=15, step_interval=1):
    window = int(time_window * 12 / step_interval) + 1

    interest = slice((window - 1) // 2, -(window - 1) // 2)
    corrected_preds_list = [all_data["pred"][..., 0]]
    for i in range(2):
        chunk_preds_list = []
        for series_id, grouped_iter in itertools.groupby(
            zip(all_data["key"], all_data["pred"][..., 1 + i]),
            key=lambda pair: pair[0].split("_")[0],
        ):
            preds_chunk_list = [preds_chunk for _, preds_chunk in grouped_iter]
            n_chunks = len(preds_chunk_list)
            grouped_preds = np.concatenate(preds_chunk_list)

            max_preds = np.max(rolling(grouped_preds, window=window, axis=-1), axis=-1)
            max_preds[~np.isclose(grouped_preds[..., interest], max_preds)] = 0
            corrected_preds = np.zeros(len(grouped_preds), dtype=grouped_preds.dtype)
            corrected_preds[interest] = max_preds
            chunk_preds_list.append(corrected_preds.reshape(n_chunks, -1))

        corrected_preds_list.append(np.concatenate(chunk_preds_list, axis=0))
    corrected_preds = np.stack(corrected_preds_list, axis=-1)
    return corrected_preds


def get_submit_df(all_data, modes: PostProcessModes = None):
    return cmi_dss_lib.utils.post_process.post_process_for_seg(
        preds=all_data["pred"],
        downsample_rate=2,
        keys=all_data["key"],
        score_th=0.005,
        distance=96,
        post_process_modes=modes,
    ).to_pandas()


def get_pred_data(i_fold):
    all_keys, all_preds, all_labels = np.load(
        project_root_path
        / "output"
        / "train"
        / "exp005-lstm-feature-2"
        / f"predicted-fold_{i_fold}.npz"
    ).values()
    all_series_ids = np.array([str(k).split("_")[0] for k in all_keys])
    all_data = npu.from_dict(
        {"key": all_keys, "pred": all_preds, "label": all_labels, "series_id": all_series_ids}
    )
    return all_data


scores_list = []
for i_fold in range(5):
    all_data = get_pred_data(i_fold)

    unique_series_ids = np.unique(all_data["series_id"])

    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")
    event_df = event_df[event_df["series_id"].isin(unique_series_ids)].dropna()

    scores = []

    print(f"{i_fold=}")
    df_submit = get_submit_df(all_data)
    prev_score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"{prev_score:.4f} -> ", end="", flush=True)
    scores.append(prev_score)

    df_submit = get_submit_df(
        all_data, modes={"sleeping_edges_as_probs": post_process_modes["sleeping_edges_as_probs"]}
    )
    score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"sleeping_edges_as_probs: {score:.4f}")
    scores.append(score)

    df_submit = get_submit_df(
        all_data,
        modes={"cutting_probs_by_sleep_prob": post_process_modes["cutting_probs_by_sleep_prob"]},
    )
    score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"cutting_probs_by_sleep_prob: {score:.4f}")
    scores.append(score)

    df_submit = get_submit_df(all_data, modes=post_process_modes)
    score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"both: {score:.4f}")
    scores.append(score)

    print()

    scores_list.append(scores)


scores = np.array(scores_list)  # (fold, mode)

print(f"mean score = {', '.join(map('{:.4f}'.format, np.mean(scores, axis=0)))}")


#

scores = np.array(
    [
        cmi_dss_lib.utils.metrics.event_detection_ap(
            event_df.query(f"series_id == '{series_id}'"),
            df_submit.query(f"series_id == '{series_id}'"),
        )
        if len(event_df.query(f"series_id == '{series_id}'")) > 0
        else np.nan
        for series_id in tqdm.tqdm(unique_series_ids)
    ]
)
assert len(unique_series_ids) == len(scores)
sel = ~np.isnan(scores)
unique_series_ids = unique_series_ids[sel]
scores = scores[sel]


i = 0

order = np.argsort(scores)
score = scores[order][i]
series_id = str(unique_series_ids[order][i])

data = all_data[all_data["series_id"] == series_id]

import cmi_dss_lib.utils.visualize

plotter = cmi_dss_lib.utils.visualize.Plotter(data, post_process_modes)
plotter.plot(series_id)
