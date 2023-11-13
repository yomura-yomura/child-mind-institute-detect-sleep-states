from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmi_dss_lib.utils.metrics import event_detection_ap
from cmi_dss_lib.utils.post_process import post_process_for_seg
from tqdm import tqdm

plt.style.use("ggplot")



def read_npz(path_data:Path)->dict:

    data = np.load(path_data)
    keys = data["key"]
    series_ids = list(set([str(k).split("_")[0] for k in keys]))
    preds = data["pred"]
    labels = data["label"]
    return {"keys":keys,"labels":labels,"preds":preds,"series_ids":series_ids}

def read_folds(path_data:Path,folds,df_event:pd.DataFrame,distance = 96,th_score=0.005,mode = None) -> list[dict]:

    dict_data = {}
    for f in tqdm(range(folds),desc = "fold:"):
        file_name = f"eda/predicted-fold_{f}.npz"
        path_file = path_data /file_name
        dict_result =read_npz(path_file)
        series_ids = dict_result["series_ids"]

        gt_df = df_event[df_event["series_id"].isin(series_ids)].dropna().reset_index(drop=True)
        df_sub = post_process_for_seg(keys=dict_result["keys"],preds = dict_result["preds"],score_th = th_score,distance=distance,post_process_modes = mode,downsample_rate=2).to_pandas()
        dict_result[f"score"] = [event_detection_ap(solution=gt_df,submission=df_sub)]
        df_score_per_id = score_per_id(sub_df = df_sub,event_df=gt_df).sort_values(by="score")
        dict_result[f"score_per_id"] = df_score_per_id 
        dict_result["submit"] = df_sub
        dict_result["event"] = gt_df
        dict_data[f"fold_{f}"] = dict_result 
        
    return dict_data

def score_folds(path_data:Path,folds,distance = 96,th_score=0.005,mode = None) -> pd.DataFrame:

    df_event = pd.read_csv(path_data / "train_events.csv")
    dict_data = {}
    list_score = []
    for f in tqdm(range(folds),desc = "fold:"):
        file_name = f"eda/predicted-fold_{f}.npz"
        path_file = path_data /file_name
        dict_result =read_npz(path_file)
        series_ids = np.unique(dict_result["series_ids"])

        gt_df = df_event[df_event["series_id"].isin(series_ids)].dropna().reset_index(drop=True)


        # same
        df_sub = post_process_for_seg(keys=dict_result["keys"],preds = dict_result["preds"],score_th = th_score,distance=distance,post_process_modes = mode).to_pandas()

        # same
        score = event_detection_ap(solution=gt_df,submission=df_sub)
        list_score.append(score)
        dict_data[f"fold_{f}"] = [score]
    mean_score = np.mean(list_score)
    dict_data["Avg"] = mean_score
    return pd.DataFrame(dict_data)


def score_per_id(sub_df:pd.DataFrame,event_df:pd.DataFrame) -> pd.DataFrame:
    sub_id = event_df["series_id"].unique()
    dict_result = {"series_id":[],"score":[]}
    for s_id in tqdm(sub_id,desc="scoreing"):
        dict_result["series_id"].append(s_id)
        dict_result["score"].append(event_detection_ap(solution=event_df[event_df["series_id"] == s_id],submission=sub_df[sub_df["series_id"] == s_id]))
    return pd.DataFrame(dict_result)

def plot_pred(dict_result,list_id:list[str] ,df_feat:pd.DataFrame,mode:Literal["all","onset","wakeup"],use_sleep:bool = True,num_chunks:int=5):
    keys = dict_result["keys"]
    preds = dict_result["preds"]
    labels = dict_result["labels"]
    df_submit = dict_result["submit"]
    # get series ids",
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    #unique_series_ids = np.unique(series_ids)\n",
    # get random series\n",
    #random_series_ids = [unique_series_ids[3]]#np.random.choice(unique_series_ids, num_samples)\n",
    #random_series_ids = np.random.choice(unique_series_ids, num_samples)\n",
    list_plot=[]
    for i, random_series_id in enumerate(list_id):
        # get random series\n",
        
        series_idx = np.where(series_ids == random_series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 3)
        this_series_labels = labels[series_idx].reshape(-1, 3)
        this_series_df = df_submit[df_submit["series_id"] == random_series_id].reset_index(drop = True)
        feat = df_feat[df_feat["series_id"] == random_series_id][["anglez","enmo"]]
        val_anglez = feat["anglez"].values
        val_enmo = feat["enmo"].values
        val_wakeup = this_series_df[this_series_df["event"] == "wakeup"]["step"].values
        val_onset = this_series_df[this_series_df["event"] == "onset"]["step"].values
        val_step = np.arange(1,preds[series_idx].reshape(-1,3).shape[0]+1)
        this_onset = np.isin(val_step,val_onset).astype(int)
        this_wakeup = np.isin(val_step,val_wakeup).astype(int)
        
        # split series\n",
        this_series_preds = np.split(this_series_preds, num_chunks)
        this_series_wakeup = np.split(this_wakeup, num_chunks)
        this_series_anglez = np.split(val_anglez,num_chunks)
        this_series_enmo = np.split(val_enmo,num_chunks)
        this_series_onset = np.split(this_onset, num_chunks)
        this_series_labels = np.split(this_series_labels, num_chunks)
        for j in range(num_chunks):
            this_series_preds_chunk = this_series_preds[j]
            this_series_labels_chunk = this_series_labels[j]
            this_series_anglez_chunk = this_series_anglez[j]
            this_series_enmo_chunk = this_series_enmo[j]
            # get onset and wakeup idx\n",
            onset_idx = np.nonzero(this_series_labels_chunk[:, 1])[0]*2
            wakeup_idx = np.nonzero(this_series_labels_chunk[:, 2])[0]*2
            pred_onset_idx = np.nonzero(this_series_onset[j])[0]
            pred_wakeup_idx = np.nonzero(this_series_wakeup[j])[0]
            fig,axs = plt.subplots(3,1,
                                   gridspec_kw=dict(height_ratios=[6,2,2], hspace=0.1),
                                   figsize = (20,10))
            #fig.tight_layout()
            if use_sleep:
                axs[0].plot(this_series_preds_chunk[:, 0], label="prop_sleep")
            if mode == "all" or mode == "onset":
                axs[0].plot(this_series_preds_chunk[:, 1], label="prop_onset")
                axs[0].vlines(pred_onset_idx, 0, 1, label="pred_label_onset", linestyles="dotted", color="C1")
                axs[0].vlines(onset_idx, 0, 1, label="actual_onset", linestyles="dashed", color="C1",linewidth=4)
            if mode == "all" or mode == "wakeup":
                axs[0].plot(this_series_preds_chunk[:, 2], label="prop_wakeup")
                axs[0].vlines(pred_wakeup_idx, 0, 1, label="pred_label_wakeup", linestyles="dotted", color="C4")
                axs[0].vlines(wakeup_idx, 0, 1, label="actual_wakeup", linestyles="dashed", color="C4",linewidth=4)
            axs[0].set_ylim(0, 1)
            axs[0].set_title(f"series_id: {random_series_id} chunk_id: {j}")
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            axs[1].set_title(f"anglez")
            axs[2].set_title(f"enmo")
            axs[1].plot(this_series_anglez_chunk,label="anglez")
            axs[1].vlines(wakeup_idx, -50, 50, label="actual_wakeup", linestyles="dashed", color="C4",linewidth=4)
            axs[1].vlines(onset_idx, -50, 50, label="actual_onset", linestyles="dashed", color="C1",linewidth=4)
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            axs[2].plot(this_series_enmo_chunk,label="enmo")
            axs[2].vlines(wakeup_idx, 0, 1, label="actual_wakeup", linestyles="dashed", color="C4",linewidth=4)
            axs[2].vlines(onset_idx, 0, 1, label="actual_onset", linestyles="dashed", color="C1",linewidth=4)
            axs[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        list_plot.append(fig)
    return list_plot