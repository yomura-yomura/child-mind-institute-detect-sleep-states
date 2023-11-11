import pathlib

import numpy as np
import numpy_utility as npu
import pandas as pd
import plotly.express as px

project_root_path = pathlib.Path(__file__).parent.parent.parent


def get_pred_data(i_fold):
    all_keys, all_preds, all_labels = np.load(
        project_root_path / "predicted" / "ranchantan" / "exp005-lstm-feature-2" / f"predicted-fold_{i_fold}.npz"
    ).values()
    all_series_ids = np.array([str(k).split("_")[0] for k in all_keys])
    all_data = npu.from_dict({"key": all_keys, "pred": all_preds, "label": all_labels, "series_id": all_series_ids})
    return all_data


data = get_pred_data(i_fold=0)


i = 1

record = data[i]

df = pd.DataFrame(record["pred"], columns=["sleep", "onset", "wakeup"]).assign(step=np.arange(record["pred"].shape[0]))


label_df = pd.DataFrame(record["label"], columns=["sleep", "onset", "wakeup"]).assign(
    step=2 * np.arange(record["label"].shape[0])
)
onset_labels = label_df[label_df["onset"].astype(bool)]["step"]
wakeup_label_df = label_df[label_df["wakeup"].astype(bool)]

df = df.melt(id_vars=["step"], var_name="type", value_name="prob")

fig = px.line(df, x="step", y="prob", color="type")
fig.show()
