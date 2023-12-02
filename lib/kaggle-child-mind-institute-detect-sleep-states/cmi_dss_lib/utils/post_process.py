import pathlib
from typing import Literal, TypeAlias, TypedDict

import cmi_dss_lib.datamodule.seg
import numpy as np
import numpy_utility as npu
import pandas as pd
import polars as pl
from nptyping import DataFrame, Float, Int, NDArray, Shape, Structure
from scipy.signal import find_peaks

SubmissionDataFrame = DataFrame[
    Structure["row_id: Int, series_id: Str, step: Int, event: Str, score: Float"]
]


class SleepingEdgesAsProbsSetting(TypedDict):
    sleep_prob_th: float
    min_sleeping_hours: int


class SleepingEdgesAsProbsSettingByEvent(TypedDict):
    onset: SleepingEdgesAsProbsSetting
    wakeup: SleepingEdgesAsProbsSetting


class CuttingProbsBySleepProbSetting(TypedDict):
    watch_interval_hour: float
    sleep_occupancy_th: float


class CuttingProbsBySleepProbSettingByEvent(TypedDict):
    onset: CuttingProbsBySleepProbSetting
    wakeup: CuttingProbsBySleepProbSetting


class CuttingProbsOnRepeating(TypedDict):
    prepare_data_dir_path: str


class PostProcessModeWithSetting(TypedDict, total=False):
    sleeping_edges_as_probs: SleepingEdgesAsProbsSetting | SleepingEdgesAsProbsSettingByEvent
    cutting_probs_by_sleep_prob: CuttingProbsBySleepProbSetting | CuttingProbsBySleepProbSettingByEvent
    cutting_probs_on_repeating: CuttingProbsOnRepeating


PostProcessModes: TypeAlias = PostProcessModeWithSetting | None


def post_process_for_seg(
    series_id: str,
    preds: NDArray[Shape["*, 3"], Float],
    labels: list[str],
    downsample_rate: int,
    score_th: float = 0.01,
    distance: int = 5000,
    width: int | None = None,
    post_process_modes: PostProcessModes = None,
    start_timing_dict: dict | None = None,
    n_records_per_series_id: int | None = None,
    print_msg: bool = False,
) -> SubmissionDataFrame:
    """make submission dataframe for segmentation task

    Args:
        series_id:
        preds: (duration, 3)
        labels:
        downsample_rate: see conf
        score_th: threshold for score. Defaults to 0.5.
        distance: minimum interval between detectable peaks
        width:
        post_process_modes: extra post process names can be given
        start_timing_dict:
        n_records_per_series_id:
        print_msg: print info
    Returns:
        pl.DataFrame: submission dataframe
    """
    assert preds.ndim == 2

    if post_process_modes is None:
        post_process_modes = {}

    possible_events = [
        cmi_dss_lib.datamodule.seg.mapping[label]
        for label in labels
        if label in cmi_dss_lib.datamodule.seg.mapping
    ]

    if "sleeping_edges_as_probs" in post_process_modes:
        if print_msg:
            print("enable 'sleeping_edges_as_probs'")
        preds = adapt_sleeping_edges_as_probs(
            preds,
            downsample_rate=downsample_rate,
            **post_process_modes["sleeping_edges_as_probs"],
        )

    if "cutting_probs_by_sleep_prob" in post_process_modes:
        if print_msg:
            print("enable 'cutting_probs_by_sleep_prob'")
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

    if "cutting_probs_on_repeating" in post_process_modes:
        if print_msg:
            print("enable 'cutting_probs_on_repeating'")
        setting = post_process_modes["cutting_probs_on_repeating"]
        for i, interval in zip(
            *get_repeating_indices_and_intervals(
                setting["prepare_data_dir_path"],
                series_id,
            )
        ):
            preds[i : i + interval] = 0

    if preds.shape[-1] == 3:
        label_index_dict = {event: i for i, event in enumerate(["sleep", "onset", "wakeup"])}
    else:
        label_index_dict = {
            event: labels.index(f"event_{event}") if f"event_{event}" in labels else None
            for event in ["sleep", "onset", "wakeup"]
        }

    if start_timing_dict is not None:
        timing = (
            pd.Series(np.arange(preds.shape[0]).astype("m8[s]") * 5) + start_timing_dict[series_id]
        )
        sel = ((15 <= timing.dt.hour) & (timing.dt.hour <= 18)).to_numpy(bool)
        preds[sel, 1] = 0

        sel = ((20 <= timing.dt.hour) | (timing.dt.hour <= 2)).to_numpy(bool)
        preds[sel, 2] = 0

    records = []
    for i, event_name in enumerate(possible_events):
        this_event_preds = preds[:, label_index_dict[event_name]]
        steps = find_peaks(this_event_preds, height=score_th, distance=distance, width=width)[0]
        scores = this_event_preds[steps]

        for step, score in zip(steps, scores, strict=True):
            if "cutting_probs_by_sleep_prob" in post_process_modes:
                n_steps = preds.shape[0]
                sleep_preds = preds[:, label_index_dict["sleep"]]

                if event_name == "onset":
                    max_step = min(step + watch_interval_hour[event_name], n_steps - 1)
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
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pd.DataFrame(records).sort_values(by=["series_id", "step"])
    if n_records_per_series_id is not None:
        sub_df = sub_df.sort_values(["score"], ascending=False).head(n_records_per_series_id)
    sub_df = sub_df[["series_id", "step", "event", "score"]]
    return sub_df


def adapt_sleeping_edges_as_probs(
    preds, downsample_rate: int, sleep_prob_th: float, min_sleeping_hours: int
) -> NDArray:
    n_steps = len(preds)
    # concat_pred = preds.reshape(-1, 3)
    _, props = find_peaks(
        preds[..., 0],
        width=min_sleeping_hours * 12 / downsample_rate * 60,
        height=sleep_prob_th,
    )

    for l_i, r_i in zip(props["left_ips"], props["right_ips"]):
        for i, pos in enumerate([l_i, r_i]):
            preds[
                int(np.floor(pos)),
                1 + i,
            ] *= (
                1
                + preds[
                    int(np.floor(pos)),
                    0,
                ]
            )

    return preds


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


def get_repeating_indices_and_intervals(
    data_dir_path: str | pathlib.Path[str],
    series_id: str,
    repeating_interval_hour: int = 24 * 60 * 60,
) -> tuple[NDArray[Shape['"*"'], Int], NDArray[Shape['"*"'], Int]]:
    repeating_interval = repeating_interval_hour // 5
    data_dir_path = pathlib.Path(data_dir_path)
    anglez_data = np.load(data_dir_path / series_id / "anglez.npy")
    enmo_data = np.load(data_dir_path / series_id / "enmo.npy")

    is_same = np.all(
        [
            np.isclose(data[repeating_interval:], data[:-repeating_interval])
            for data in [anglez_data, enmo_data]
        ],
        axis=0,
    )

    indices_at_same = np.where(is_same)[0]
    if len(indices_at_same) == 0:
        return np.array([], dtype="i8"), np.array([], dtype="i8")
    intervals = (
        np.array(
            list(
                map(
                    len,
                    "".join(
                        (indices_at_same[1:] - indices_at_same[:-1] > 1).astype(int).astype(str)
                    ).split("1"),
                )
            )
        )
        + 1
    )
    assert np.sum(intervals) == len(indices_at_same)
    start_indices_at_same = indices_at_same[[0, *np.cumsum(intervals[:-1])]]

    # validation
    _start_prev_indices_at_same = start_indices_at_same - 1
    _start_prev_indices_at_same = _start_prev_indices_at_same[_start_prev_indices_at_same >= 0]
    assert np.all(is_same[_start_prev_indices_at_same] == np.False_)
    assert np.all(is_same[start_indices_at_same] == np.True_)
    assert np.all(is_same[start_indices_at_same + intervals - 1] == np.True_)
    _start_next_indices_at_same = start_indices_at_same + intervals
    _start_next_indices_at_same = _start_next_indices_at_same[
        _start_next_indices_at_same < len(is_same)
    ]
    assert np.all(is_same[_start_next_indices_at_same] == np.False_)
    return repeating_interval + start_indices_at_same, intervals
