import itertools
import pathlib
from typing import Literal

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import matplotlib.pyplot as plt
import numpy as np
import numpy_utility as npu
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset
from child_mind_institute_detect_sleep_states.data.comp_dataset import rolling

project_root_path = pathlib.Path(__file__).parent.parent


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


def get_submit_df(all_data, modes=None):
    return cmi_dss_lib.utils.post_process.post_process_for_seg(
        #
        preds=all_data["pred"],
        # preds=corrected_preds[:, :, [1, 2]],
        keys=all_data["key"],
        score_th=0.005,
        distance=96,
        post_process_modes=modes,
    ).to_pandas()


scores_list = []
for i_fold in range(5):
    all_keys, all_preds, all_labels = np.load(
        project_root_path
        / "output"
        / "train"
        / "exp005-lstm-feature-2"
        # / "preds"
        # / "uchida-1"
        / f"predicted-fold_{i_fold}.npz"
    ).values()
    all_series_ids = np.array([str(k).split("_")[0] for k in all_keys])
    all_data = npu.from_dict(
        {"key": all_keys, "pred": all_preds, "label": all_labels, "series_id": all_series_ids}
    )

    unique_series_ids = np.unique(all_series_ids)

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

    df_submit = get_submit_df(all_data, modes=["sleeping_edges_as_probs"])
    score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"sleeping_edges_as_probs: {score:.4f}")
    scores.append(score)

    df_submit = get_submit_df(all_data, modes=["cutting_probs_by_sleep_prob"])
    score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"cutting_probs_by_sleep_prob: {score:.4f}")
    scores.append(score)

    df_submit = get_submit_df(all_data, modes=["sleeping_edges_as_probs", "cutting_probs_by_sleep_prob"])
    score = cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )
    print(f"both: {score:.4f}")
    scores.append(score)

    print()

    scores_list.append(scores)


scores = np.array(scores_list)  # (fold, mode)

print(f"mean score = {', '.join(map('{:.4f}'.format, np.mean(scores, axis=0)))}")

fads


class Plotter:
    def __init__(self, data):
        self.series_ids = np.array(list(map(lambda x: x.split("_")[0], data["key"])))
        self.keys = data["key"]
        self.preds = data["pred"]
        self.labels = data["label"]
        self.df_submit = get_submit_df(data)

        unique_series_ids = np.unique(self.series_ids)

        self.score = cmi_dss_lib.utils.metrics.event_detection_ap(
            event_df[event_df["series_id"].isin(unique_series_ids)], self.df_submit
        )
        print(f"{self.score = :.4f}")

    def plot(
        self,
        series_id: str,
        num_chunks: int = 5,
        mode: Literal["all", "onset", "wakeup"] = "all",
        use_sleep: bool = True,
    ):
        series_idx = np.where(self.series_ids == series_id)[0]
        this_series_preds = self.preds[series_idx].reshape(-1, 3)
        this_series_labels = self.labels[series_idx].reshape(-1, 3)
        this_series_df = self.df_submit[self.df_submit["series_id"] == series_id].reset_index(
            drop=True
        )
        val_wakeup = this_series_df[this_series_df["event"] == "wakeup"]["step"].values
        val_onset = this_series_df[this_series_df["event"] == "onset"]["step"].values
        val_step = np.arange(1, self.preds[series_idx].reshape(-1, 3).shape[0] + 1)
        this_onset = np.isin(val_step, val_onset).astype(int)
        this_wakeup = np.isin(val_step, val_wakeup).astype(int)

        # split series
        this_series_preds = np.split(this_series_preds, num_chunks)
        this_series_wakeup = np.split(this_wakeup, num_chunks)
        this_series_onset = np.split(this_onset, num_chunks)
        this_series_labels = np.split(this_series_labels, num_chunks)

        fig, axs = plt.subplots(num_chunks, 1, figsize=(20, 5 * num_chunks))
        if num_chunks == 1:
            axs = [axs]
        for j in range(num_chunks):
            this_series_preds_chunk = this_series_preds[j]
            this_series_labels_chunk = this_series_labels[j]

            # get onset and wakeup idx
            onset_idx = np.nonzero(this_series_labels_chunk[:, 1])[0] * 2
            wakeup_idx = np.nonzero(this_series_labels_chunk[:, 2])[0] * 2

            pred_onset_idx = np.nonzero(this_series_onset[j])[0]
            pred_wakeup_idx = np.nonzero(this_series_wakeup[j])[0]

            if use_sleep:
                axs[j].plot(this_series_preds_chunk[:, 0], label="pred_sleep")
            if mode == "all" or mode == "onset":
                axs[j].plot(this_series_preds_chunk[:, 1], label="pred_onset")
                axs[j].vlines(
                    pred_onset_idx, 0, 1, label="pred_label_onset", linestyles="dotted", color="C1"
                )
                axs[j].vlines(
                    onset_idx,
                    0,
                    1,
                    label="actual_onset",
                    linestyles="dashed",
                    color="C1",
                    linewidth=4,
                )
            if mode == "all" or mode == "wakeup":
                axs[j].plot(this_series_preds_chunk[:, 2], label="pred_wakeup")
                axs[j].vlines(
                    pred_wakeup_idx,
                    0,
                    1,
                    label="pred_label_wakeup",
                    linestyles="dotted",
                    color="C4",
                )
                axs[j].vlines(
                    wakeup_idx,
                    0,
                    1,
                    label="actual_wakeup",
                    linestyles="dashed",
                    color="C4",
                    linewidth=4,
                )

            axs[j].set_ylim(0, 1)
            axs[j].set_title(f"series_id: {series_id} chunk_id: {j}")
            axs[j].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()
        plt.show()


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

plotter = Plotter(data)
plotter.plot(series_id)
