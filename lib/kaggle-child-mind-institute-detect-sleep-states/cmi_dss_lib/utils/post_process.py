import numpy as np
import numpy_utility as npu
import polars as pl
from numpy.typing import NDArray
from scipy.signal import find_peaks


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.01,
    distance: int = 5000,
    post_process_modes: list[str] | None = None,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 3)
        score_th (float, optional): threshold for score. Defaults to 0.5.
        distance: minimum interval between detectable peaks
        post_process_modes: extra post process names can be given
    Returns:
        pl.DataFrame: submission dataframe
    """
    around_hour = 6
    th_hour_step = around_hour*60*5
    cut_prob_th = 0.5
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    if post_process_modes is not None:
        if "adapt_sleep_prob" in post_process_modes:
            print("enable 'adapt_sleep_prob'")
            data = adapt_sleep_prob(
                npu.from_dict(
                    {
                        "key": keys,
                        "pred": preds,
                        "series_id": series_ids,
                    }
                )
            )
            keys = data["key"]
            preds = data["pred"]
            series_id = data["series_id"]

            series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
     
    preds = preds if "cut_sleep_prob" in post_process_modes else  preds[:, :, [1, 2]]

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3) if "cut_sleep_prob" in post_process_modes else preds[series_idx].reshape(-1, 2)

        if "cut_sleep_prob" in post_process_modes:
            max_len_step = this_series_preds.shape[0]
            sleep_preds = this_series_preds[:, 0]

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i+1] if "cut_sleep_prob" in post_process_modes else this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                if "cut_sleep_prob" in post_process_modes:
                    if event_name == "onset":
                        max_step = step+th_hour_step if step+th_hour_step <= max_len_step-1 else max_len_step-1
                        sleep_score = np.median(sleep_preds[step:max_step] )

                    if event_name == "wakeup":
                        min_step = step-th_hour_step if step-th_hour_step >=0 else step
                        sleep_score = np.median(sleep_preds[min_step:step] )

                    # skip
                    if sleep_score < cut_prob_th:
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

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


def adapt_sleep_prob(data: NDArray) -> NDArray:
    duration = data["pred"].shape[1]

    corrected_data_list = []
    for series_id, grouped_data in npu.groupby(data, "series_id"):
        grouped_data = grouped_data[np.argsort(grouped_data["key"])]
        n = len(grouped_data)
        concat_pred = grouped_data["pred"].reshape(-1, 3)
        # partial_preds = concat_pred[..., 0][:20_000]
        _, props = find_peaks(concat_pred[..., 0], width=6 * 12 / 2 * 60, height=0.2)

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
        corrected_data["pred"] = concat_pred.reshape(n, duration, 3)
        corrected_data_list.append(corrected_data)

    return np.concatenate(corrected_data_list)
