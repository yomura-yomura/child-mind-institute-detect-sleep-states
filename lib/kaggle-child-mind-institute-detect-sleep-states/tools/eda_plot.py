import pandas as pd
from pathlib import Path
from cmi_dss_lib.utils.metrics import event_detection_ap
from cmi_dss_lib.utils.post_process import post_process_for_seg
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
chunk = 4
plt.style.use("ggplot")

RESULT_DIR = "../../../data/eda/predicted-fold_0.npz"
def plot_random_sample(keys,preds, labels,list_id:list[str] ,df_submit,mode:Literal["all","onset","wakeup"],use_sleep:bool = True,num_chunks:int=5):
    # get series ids",
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys))),
    #unique_series_ids = np.unique(series_ids)\n",
    # get random series\n",
    #random_series_ids = [unique_series_ids[3]]#np.random.choice(unique_series_ids, num_samples)\n",
    #random_series_ids = np.random.choice(unique_series_ids, num_samples)\n",
    for i, random_series_id in enumerate(list_id):
        # get random series\n",
        series_idx = np.where(series_ids == random_series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)
        this_series_labels = labels[series_idx].reshape(-1, 3)
        this_series_df = df_submit[df_submit["series_id"] == random_series_id].reset_index(drop = True)
        val_wakeup = this_series_df[this_series_df["event"] == "wakeup"]["step"].values
        val_onset = this_series_df[this_series_df["event"] == "onset"]["step"].values
        val_step = np.arange(1,preds[series_idx].reshape(-1,3).shape[0]+1)
        this_onset = np.isin(val_step,val_onset).astype(int)
        this_wakeup = np.isin(val_step,val_wakeup).astype(int)
        
        # split series\n",
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
            # get onset and wakeup idx\n",
            onset_idx = np.nonzero(this_series_labels_chunk[:, 1])[0]*2
            wakeup_idx = np.nonzero(this_series_labels_chunk[:, 2])[0]*2
            pred_onset_idx = np.nonzero(this_series_onset[j])[0]
            pred_wakeup_idx = np.nonzero(this_series_wakeup[j])[0]
            if use_sleep:
                axs[j].plot(this_series_preds_chunk[:, 0], label="pred_sleep")
            if mode == "all" or mode == "onset":
                axs[j].plot(this_series_preds_chunk[:, 1], label="pred_onset")
                axs[j].vlines(pred_onset_idx, 0, 1, label="pred_label_onset", linestyles="dotted", color="C1")
                axs[j].vlines(onset_idx, 0, 1, label="actual_onset", linestyles="dashed", color="C1",linewidth=4)
            if mode == "all" or mode == "wakeup":
                axs[j].plot(this_series_preds_chunk[:, 2], label="pred_wakeup")
                axs[j].vlines(pred_wakeup_idx, 0, 1, label="pred_label_wakeup", linestyles="dotted", color="C4")
                axs[j].vlines(wakeup_idx, 0, 1, label="actual_wakeup", linestyles="dashed", color="C4",linewidth=4)
            axs[j].set_ylim(0, 1)
            axs[j].set_title(f"series_id: {random_series_id} chunk_id: {j}")
            axs[j].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    return fig