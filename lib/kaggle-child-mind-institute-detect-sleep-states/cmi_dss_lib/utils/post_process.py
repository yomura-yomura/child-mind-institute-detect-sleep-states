from typing import Sequence, TypeAlias, TypedDict

import numpy as np
import numpy_utility as npu
import pandas as pd
import polars as pl
from nptyping import DataFrame, Float, NDArray, Shape, Structure
from scipy.signal import find_peaks

SubmissionDataFrame = DataFrame[
    Structure["row_id: Int, series_id: Str, step: Int, event: Str, score: Float"]
]


class SleepingEdgesAsProbsSetting(TypedDict):
    sleep_prob_th: float
    min_sleeping_hours: int


class CuttingProbsBySleepProbSetting(TypedDict):
    watch_interval_hour: int
    sleep_occupancy_th: float


class PostProcessModeWithSetting(TypedDict, total=False):
    sleeping_edges_as_probs: SleepingEdgesAsProbsSetting
    cutting_probs_by_sleep_prob: CuttingProbsBySleepProbSetting


PostProcessModes: TypeAlias = PostProcessModeWithSetting | None


def post_process_for_seg(
    keys: Sequence[str],
    preds: NDArray[Shape["*, 3"], Float],
    downsample_rate: int,
    score_th: float = 0.01,
    distance: int = 5000,
    post_process_modes: PostProcessModes = None,
    print_msg: bool = False,
) -> SubmissionDataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys: list of keys. key is "{series_id}_{chunk_id}"
        preds: (num_series * num_chunks, duration, 3)
        downsample_rate: see conf
        score_th: threshold for score. Defaults to 0.5.
        distance: minimum interval between detectable peaks
        post_process_modes: extra post process names can be given
        print_msg: print info
    Returns:
        pl.DataFrame: submission dataframe
    """

    if post_process_modes is None:
        post_process_modes = {}

    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    if "sleeping_edges_as_probs" in post_process_modes:
        if print_msg:
            print("enable 'sleeping_edges_as_probs'")
        data = adapt_sleeping_edges_as_probs(
            npu.from_dict(
                {
                    "key": keys,
                    # "pred": preds,
                    "pred": preds.astype("f4"),
                    "series_id": series_ids,
                }
            ),
            downsample_rate=downsample_rate,
            **post_process_modes["sleeping_edges_as_probs"],
        )
        keys = data["key"]
        preds = data["pred"]

        series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))

    # preds = preds if "cutting_probs_by_sleep_prob" in post_process_modes else preds[:, :, [1, 2]]
    if "cutting_probs_by_sleep_prob" in post_process_modes:
        if print_msg:
            print("enable 'cutting_probs_by_sleep_prob'")
        setting = post_process_modes["cutting_probs_by_sleep_prob"]
        sleep_occupancy_th = setting["sleep_occupancy_th"]
        th_hour_step = setting["watch_interval_hour"] * 60 * 12 // downsample_rate
    else:
        sleep_occupancy_th = th_hour_step = None

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i + 1]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores, strict=True):
                if "cutting_probs_by_sleep_prob" in post_process_modes:
                    max_len_step = this_series_preds.shape[0]
                    sleep_preds = this_series_preds[:, 0]

                    if event_name == "onset":
                        max_step = min(step + th_hour_step, max_len_step - 1)
                        if step < max_step:
                            sleep_score = np.median(sleep_preds[step:max_step])
                        else:
                            sleep_score = np.nan
                    elif event_name == "wakeup":
                        min_step = max(step - th_hour_step, 0)
                        if min_step < step:
                            sleep_score = np.median(sleep_preds[min_step:step])
                        else:
                            sleep_score = np.nan
                    else:
                        assert False

                    # skip
                    if sleep_score < sleep_occupancy_th:
                        continue

                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": unique_series_ids[-1],
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pd.DataFrame(records).sort_values(by=["series_id", "step"])
    sub_df["row_id"] = np.arange(len(sub_df))
    sub_df = sub_df[["row_id", "series_id", "step", "event", "score"]]
    return sub_df


def adapt_sleeping_edges_as_probs(
    data: NDArray, downsample_rate: int, sleep_prob_th: float, min_sleeping_hours: int
) -> NDArray:
    # duration = data["pred"].shape[-2]

    corrected_data_list = []
    for series_id, grouped_data in npu.groupby(data, "series_id"):
        grouped_data = grouped_data[np.argsort(grouped_data["key"])]
        n = len(grouped_data)
        concat_pred = grouped_data["pred"].reshape(-1, 3)
        # partial_preds = concat_pred[..., 0][:20_000]
        _, props = find_peaks(
            concat_pred[..., 0],
            width=min_sleeping_hours * 12 / downsample_rate * 60,
            height=sleep_prob_th,
        )

        # fig = px.line(y=concat_pred[..., 0])
        # for l_i, r_i in zip(props["left_ips"], props["right_ips"]):
        #     fig.add_vrect(x0=l_i, x1=r_i, line_width=0, fillcolor="red", opacity=0.2)
        # fig.show()

        for l_i, r_i in zip(props["left_ips"], props["right_ips"]):
            # interval = 12 // 2 * 60
            # interval = 12 // 2 * 10
            for i, pos in enumerate([l_i, r_i]):
                # interest = slice(
                #     round(pos) - interval,
                #     round(pos) + interval,
                # )
                concat_pred[
                    int(np.floor(pos)),
                    # interest,
                    1 + i,
                ] *= (
                    1
                    + concat_pred[
                        int(np.floor(pos)),
                        # interest,
                        0,
                    ]
                )
        corrected_data = grouped_data.copy()
        corrected_data["pred"] = concat_pred.reshape(n, 3)
        corrected_data_list.append(corrected_data)

    return np.concatenate(corrected_data_list)
