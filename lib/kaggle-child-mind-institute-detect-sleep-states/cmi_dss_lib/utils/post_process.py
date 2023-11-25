from typing import Literal, Sequence, TypeAlias, TypedDict

import cmi_dss_lib.datamodule.seg
import numpy as np
import numpy_utility as npu
import pandas as pd
import polars as pl
from nptyping import DataFrame, Float, NDArray, Shape, Structure
from scipy.signal import find_peaks
from typing_extensions import NotRequired

SubmissionDataFrame = DataFrame[
    Structure["row_id: Int, series_id: Str, step: Int, event: Str, score: Float"]
]


class SleepingEdgesAsProbsSetting(TypedDict):
    sleep_prob_th: float
    min_sleeping_hours: int


class CuttingProbsBySleepProbSetting(TypedDict):
    watch_interval_hour: float
    sleep_occupancy_th: float


class PostProcessModeWithSetting(TypedDict, total=False):
    sleeping_edges_as_probs: SleepingEdgesAsProbsSetting | dict[
        Literal["onset", "wakeup"], SleepingEdgesAsProbsSetting
    ]

    cutting_probs_by_sleep_prob: CuttingProbsBySleepProbSetting | dict[
        Literal["onset", "wakeup"], CuttingProbsBySleepProbSetting
    ]


PostProcessModes: TypeAlias = PostProcessModeWithSetting | None


def post_process_for_seg(
    keys: Sequence[str],
    preds: NDArray[Shape["*, 3"], Float],
    labels: list[str],
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

    possible_events = [
        cmi_dss_lib.datamodule.seg.mapping[label]
        for label in labels
        if label in cmi_dss_lib.datamodule.seg.mapping
    ]

    if "sleeping_edges_as_probs" in post_process_modes:
        if print_msg:
            print("enable 'sleeping_edges_as_probs'")
        data = adapt_sleeping_edges_as_probs(
            keys,
            preds,
            downsample_rate=downsample_rate,
            **post_process_modes["sleeping_edges_as_probs"],
        )
        keys = data["key"]
        preds = data["pred"]

        series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))

    if "cutting_probs_by_sleep_prob" in post_process_modes:
        if print_msg:
            print("enable 'cutting_probs_by_sleep_prob'")
        # data = adapt_cutting_probs_by_sleep_prob(
        #     keys,
        #     preds,
        #     downsample_rate=downsample_rate,
        #     **post_process_modes["cutting_probs_by_sleep_prob"],
        # )
        # keys = data["key"]
        # preds = data["pred"]
        #
        # series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
        setting = post_process_modes["cutting_probs_by_sleep_prob"]
        if "onset" in setting and "wakeup" in setting:
            sleep_occupancy_th = {
                event: setting[event]["sleep_occupancy_th"] for event in ["onset", "wakeup"]
            }
            watch_interval_hour = {
                event: int(setting[event]["watch_interval_hour"] * 60 * 12 / downsample_rate)
                for event in ["onset", "wakeup"]
            }
        else:
            sleep_occupancy_th = {
                event: setting["sleep_occupancy_th"] for event in ["onset", "wakeup"]
            }
            watch_interval_hour = {
                event: int(setting["watch_interval_hour"] * 60 * 12 / downsample_rate)
                for event in ["onset", "wakeup"]
            }
    else:
        sleep_occupancy_th = watch_interval_hour = None

    if preds.shape[-1] == 3:
        label_index_dict = {event: i for i, event in enumerate(["sleep", "onset", "wakeup"])}
    else:
        label_index_dict = {
            event: labels.index("sleep") if "sleep" in labels else None
            for event in ["sleep", "onset", "wakeup"]
        }

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, len(labels))

        for i, event_name in enumerate(possible_events):
            this_event_preds = this_series_preds[:, label_index_dict[event_name]]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores, strict=True):
                if "cutting_probs_by_sleep_prob" in post_process_modes:
                    max_len_step = this_series_preds.shape[0]
                    sleep_preds = this_series_preds[:, label_index_dict["sleep"]]

                    if event_name == "onset":
                        max_step = min(step + watch_interval_hour[event_name], max_len_step - 1)
                        if step < max_step:
                            sleep_score = np.median(sleep_preds[step:max_step])
                        else:
                            sleep_score = np.nan
                    elif event_name == "wakeup":
                        min_step = max(step - watch_interval_hour[event_name], 0)
                        if min_step < step:
                            sleep_score = np.median(sleep_preds[min_step:step])
                        else:
                            sleep_score = np.nan
                    else:
                        assert False

                    # skip
                    if sleep_score < sleep_occupancy_th[event_name]:
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
    keys, preds, downsample_rate: int, sleep_prob_th: float, min_sleeping_hours: int
) -> NDArray:
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    data = npu.from_dict(
        {
            "key": keys,
            "pred": preds,
            "series_id": series_ids,
        }
    )
    corrected_data_list = []
    for series_id, grouped_data in npu.groupby(data, "series_id"):
        grouped_data = grouped_data[np.argsort(grouped_data["key"])]
        n = len(grouped_data)
        concat_pred = grouped_data["pred"].reshape(-1, 3)
        _, props = find_peaks(
            concat_pred[..., 0],
            width=min_sleeping_hours * 12 / downsample_rate * 60,
            height=sleep_prob_th,
        )

        for l_i, r_i in zip(props["left_ips"], props["right_ips"]):
            for i, pos in enumerate([l_i, r_i]):
                concat_pred[
                    int(np.floor(pos)),
                    1 + i,
                ] *= (
                    1
                    + concat_pred[
                        int(np.floor(pos)),
                        0,
                    ]
                )
        corrected_data = grouped_data.copy()
        corrected_data["pred"] = concat_pred.reshape(n, 3)
        corrected_data_list.append(corrected_data)

    return np.concatenate(corrected_data_list)


def adapt_cutting_probs_by_sleep_prob(
    keys, preds, downsample_rate: int, sleep_occupancy_th: float, watch_interval_hour: int
):
    watch_interval_hour = int(watch_interval_hour * 60 * 12 / downsample_rate)

    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    data = npu.from_dict(
        {
            "key": keys,
            "pred": preds,
            "series_id": series_ids,
        }
    )

    from child_mind_institute_detect_sleep_states.data.comp_dataset import rolling

    corrected_data_list = []
    for series_id, grouped_data in npu.groupby(data, "series_id"):
        grouped_data = grouped_data[np.argsort(grouped_data["key"])]
        n = len(grouped_data)
        concat_pred = grouped_data["pred"].reshape(-1, 3)

        median_sleep_probs = np.median(
            rolling(concat_pred[:, 0], window=watch_interval_hour, axis=0), axis=1
        )
        n_invalid_steps = len(concat_pred) - len(median_sleep_probs)

        # onset
        is_over_th = np.r_[median_sleep_probs > sleep_occupancy_th, [False] * n_invalid_steps]

        concat_pred[~is_over_th, 1] = 0
        # wakeup
        is_over_th = np.r_[[False] * n_invalid_steps, median_sleep_probs > sleep_occupancy_th]
        concat_pred[~is_over_th, 2] = 0

        corrected_data = grouped_data.copy()
        corrected_data["pred"] = concat_pred.reshape(n, 3)
        corrected_data_list.append(corrected_data)

    return np.concatenate(corrected_data_list)
