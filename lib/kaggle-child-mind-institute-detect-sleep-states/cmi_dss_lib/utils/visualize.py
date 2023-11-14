from typing import Literal

import cmi_dss_lib.utils.post_process
import matplotlib.pyplot as plt
import numpy as np
from cmi_dss_lib.utils.post_process import PostProcessModes
from numpy.typing import NDArray

import child_mind_institute_detect_sleep_states.data.comp_dataset


class Plotter:
    def __init__(self, data: NDArray, post_process_modes: PostProcessModes):
        self.series_ids = np.array(list(map(lambda x: x.split("_")[0], data["key"])))
        self.keys = data["key"]
        self.preds = data["pred"]
        self.labels = data["label"]
        self.df_submit = cmi_dss_lib.utils.post_process.post_process_for_seg(
            preds=data["pred"],
            downsample_rate=2,
            keys=data["key"],
            score_th=0.005,
            distance=96,
            post_process_modes=post_process_modes,
        )

        event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")
        unique_series_ids = np.unique(self.series_ids)

        self.score = cmi_dss_lib.utils.metrics.event_detection_ap(
            event_df[event_df["series_id"].isin(unique_series_ids)].dropna(), self.df_submit
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
        this_series_df = self.df_submit[self.df_submit["series_id"] == series_id].reset_index(drop=True)
        val_wakeup = this_series_df[this_series_df["event"] == "wakeup"]["step"].values
        val_onset = this_series_df[this_series_df["event"] == "onset"]["step"].values
        val_step = np.arange(1, self.preds[series_idx].reshape(-1, 3).shape[0] + 1)
        this_onset = np.isin(val_step, val_onset).astype(int)
        this_wakeup = np.isin(val_step, val_wakeup).astype(int)

        print(this_series_preds.shape)

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
                    pred_onset_idx,
                    0,
                    1,
                    label="pred_label_onset",
                    # linestyles="dotted",
                    color="C1",
                    alpha=0.4,
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
                    # linestyles="dotted",
                    color="C4",
                    alpha=0.4,
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
