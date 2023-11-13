from typing import Literal

import cmi_dss_lib.utils.metrics
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
        ).to_pandas()

        event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")
        unique_series_ids = np.unique(self.series_ids)
        self.score = cmi_dss_lib.utils.metrics.event_detection_ap(
            event_df[event_df["series_id"].isin(unique_series_ids)].dropna(), self.df_submit
        )
        print(f"{self.score = :.4f}")
        feat_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df("train")
        self.feat=feat_df[feat_df["series_id"].isin(unique_series_ids)]

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
        val_anglez = self.feat[self.feat["series_id"] == series_id]["anglez"].values
        val_enmo = self.feat[self.feat["series_id"] == series_id]["enmo"].values
        val_step = np.arange(1, self.preds[series_idx].reshape(-1, 3).shape[0] + 1)
        this_onset = np.isin(val_step, val_onset).astype(int)
        this_wakeup = np.isin(val_step, val_wakeup).astype(int)

        print(this_series_preds.shape)

        # split series
        this_series_preds = np.split(this_series_preds, num_chunks)
        this_series_wakeup = np.split(this_wakeup, num_chunks)
        this_series_onset = np.split(this_onset, num_chunks)
        this_series_labels = np.split(this_series_labels, num_chunks)
        this_series_anglez = np.split(val_anglez,num_chunks)
        this_series_enmo = np.split(val_enmo,num_chunks)

        for j in range(num_chunks):
            this_series_preds_chunk = this_series_preds[j]
            this_series_labels_chunk = this_series_labels[j]
            this_series_anglez_chunk = this_series_anglez[j]
            this_series_enmo_chunk = this_series_enmo[j]

            # get onset and wakeup idx
            onset_idx = np.nonzero(this_series_labels_chunk[:, 1])[0] * 2
            wakeup_idx = np.nonzero(this_series_labels_chunk[:, 2])[0] * 2

            pred_onset_idx = np.nonzero(this_series_onset[j])[0]
            pred_wakeup_idx = np.nonzero(this_series_wakeup[j])[0]

            fig,axs = plt.subplots(3,1,
                                   gridspec_kw=dict(height_ratios=[6,2,2], hspace=0.1),
                                   figsize = (20,10))
            if use_sleep:
                axs[0].plot(this_series_preds_chunk[:, 0], label="pred_sleep")

            if mode == "all" or mode == "onset":
                axs[0].plot(this_series_preds_chunk[:, 1], label="pred_onset")
                axs[0].vlines(
                    pred_onset_idx,
                    0,
                    1,
                    label="pred_label_onset",
                    # linestyles="dotted",
                    color="C1",
                    alpha=0.4,
                )
                axs[0].vlines(
                    onset_idx,
                    0,
                    1,
                    label="actual_onset",
                    linestyles="dashed",
                    color="C1",
                    linewidth=4,
                )
            if mode == "all" or mode == "wakeup":
                axs[0].plot(this_series_preds_chunk[:, 2], label="pred_wakeup")
                axs[0].vlines(
                    pred_wakeup_idx,
                    0,
                    1,
                    label="pred_label_wakeup",
                    # linestyles="dotted",
                    color="C4",
                    alpha=0.4,
                )
                axs[0].vlines(
                    wakeup_idx,
                    0,
                    1,
                    label="actual_wakeup",
                    linestyles="dashed",
                    color="C4",
                    linewidth=4,
                )

            axs[0].set_ylim(0, 1)
            axs[0].set_title(f"series_id: {series_id} chunk_id: {j}")
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            axs[1].plot(this_series_anglez_chunk,label="anglez")
            axs[1].vlines(wakeup_idx, -50, 50, label="actual_wakeup", linestyles="dashed", color="C4",linewidth=4)
            axs[1].vlines(onset_idx, -50, 50, label="actual_onset", linestyles="dashed", color="C1",linewidth=4)
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            axs[2].plot(this_series_enmo_chunk,label="enmo")
            axs[2].vlines(wakeup_idx, 0, 1, label="actual_wakeup", linestyles="dashed", color="C4",linewidth=4)
            axs[2].vlines(onset_idx, 0, 1, label="actual_onset", linestyles="dashed", color="C1",linewidth=4)
            axs[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()
        plt.show()
