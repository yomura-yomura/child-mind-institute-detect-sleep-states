import pathlib

import cmi_dss_lib.datamodule.seg
import numpy as np
import numpy_utility as npu
import pandas as pd
import plotly.express as px

project_root_path = pathlib.Path(__file__).parent.parent.parent


target_pred_dir_path = project_root_path / "predicted" / "ranchantan" / "exp005-lstm-feature-2"


def get_pred_data(i_fold):
    all_keys, all_preds, all_labels = np.load(
        target_pred_dir_path / f"predicted-fold_{i_fold}.npz"
    ).values()
    all_series_ids = np.array([str(k).split("_")[0] for k in all_keys])
    all_data = npu.from_dict(
        {"key": all_keys, "pred": all_preds, "label": all_labels, "series_id": all_series_ids}
    )
    return all_data


data = get_pred_data(i_fold=0)

import hydra

hydra.initialize(config_path="../conf", version_base="1.2")

cfg = hydra.compose("train")

datamodule = cmi_dss_lib.datamodule.seg.SegDataModule(cfg)
datamodule.setup("valid")
val_dataset = datamodule.val_dataloader().dataset

i = 1

feat_record = val_dataset[i]
pred_record = data[i]
assert pred_record["series_id"] == feat_record["key"].split("_")[0]
assert int(feat_record["key"].split("_")[1]) == i

# pred

pred_df = pd.DataFrame(pred_record["pred"], columns=["sleep", "onset", "wakeup"]).assign(
    step=np.arange(pred_record["pred"].shape[0])
)


label_df = pd.DataFrame(pred_record["label"], columns=["sleep", "onset", "wakeup"]).assign(
    step=2 * np.arange(pred_record["label"].shape[0])
)
onset_label_steps = label_df[label_df["onset"].astype(bool)]["step"].to_numpy()
wakeup_label_steps = label_df[label_df["wakeup"].astype(bool)]["step"].to_numpy()

pred_df = pred_df.melt(id_vars=["step"], var_name="type", value_name="prob")

# feat

feat_df = pd.DataFrame(feat_record["feature"].T, columns=cfg.features).assign(
    step=np.arange(feat_record["feature"].shape[1])
)
feat_df = feat_df.melt(id_vars=["step"], var_name="type", value_name="value")
fig = px.line(feat_df, x="step", y="value", color="type")
fig.show()

# fig

fig = px.line(pred_df, x="step", y="prob", color="type")
for step in onset_label_steps:
    fig.add_vline(
        x=step,
        annotation_text="onset",
        line=dict(dash="dash", color="red"),
    )
for step in wakeup_label_steps:
    fig.add_vline(
        x=step,
        annotation_text="wakeup",
        line=dict(dash="dash", color="green"),
    )
fig.show()
