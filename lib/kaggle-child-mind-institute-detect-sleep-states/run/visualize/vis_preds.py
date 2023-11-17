import pathlib

import cmi_dss_lib.datamodule.seg
import numpy as np
import numpy_utility as npu
import pandas as pd
import plotly.express as px
from cmi_dss_lib.config import TrainConfig

project_root_path = pathlib.Path(__file__).parent.parent.parent


exp_name = "exp041"

target_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan" / exp_name / "train"
# target_pred_dir_path = (
#     project_root_path / "run" / "tmp" / "predicted" / "ranchantan" / exp_name / "train"
# )
assert target_pred_dir_path.exists()

overrides_yaml_path = (
    project_root_path / "output" / "train" / exp_name / "fold_0" / ".hydra" / "overrides.yaml"
)
assert overrides_yaml_path.exists()

# def get_pred_data(i_fold):
#     all_keys, all_preds, all_labels = np.load(
#         target_pred_dir_path / f"predicted-fold_{i_fold}.npz"
#     ).values()
#     all_series_ids = np.array([str(k).split("_")[0] for k in all_keys])
#     all_data = npu.from_dict(
#         {"key": all_keys, "pred": all_preds, "label": all_labels, "series_id": all_series_ids}
#     )
#     return all_data
#
#
# data = get_pred_data(i_fold=0)

import hydra
import omegaconf
from cmi_dss_lib.datamodule.seg import Indexer

hydra.initialize(config_path="../conf", version_base="1.2")

cfg: TrainConfig = hydra.compose("train", overrides=omegaconf.OmegaConf.load(overrides_yaml_path))
#
# cfg.prev_margin_steps = 6 * 12 * 60
# cfg.next_margin_steps = 6 * 12 * 60

datamodule = cmi_dss_lib.datamodule.seg.SegDataModule(cfg)
datamodule.setup("valid")
val_dataset = datamodule.val_dataloader().dataset


def plot(i):
    feat_record = val_dataset[i]

    series_id, i = feat_record["key"].split("_")
    i = int(i)
    # assert int(feat_record["key"].split("_")[1]) == i

    preds = np.load(target_pred_dir_path / cfg.split.name / f"{series_id}.npz")["arr_0"]
    indexer = Indexer(preds.shape[0], cfg.duration, cfg.prev_margin_steps, cfg.next_margin_steps)
    start, end = indexer.get_cropping_range(i)
    preds = preds[start:end]

    # pred

    pred_df = pd.DataFrame(preds, columns=["sleep", "onset", "wakeup"]).assign(
        step=np.arange(preds.shape[0])
    )

    label_df = pd.DataFrame(feat_record["label"], columns=["sleep", "onset", "wakeup"]).assign(
        step=2 * np.arange(feat_record["label"].shape[0])
    )
    onset_label_steps = label_df[label_df["onset"].astype(bool)]["step"].to_numpy()
    wakeup_label_steps = label_df[label_df["wakeup"].astype(bool)]["step"].to_numpy()

    pred_df = pred_df.melt(id_vars=["step"], var_name="type", value_name="prob")

    # fig

    fig = px.line(pred_df, x="step", y="prob", color="type")

    feat_df = pd.DataFrame(feat_record["feature"].T, columns=cfg.features).assign(
        step=np.arange(feat_record["feature"].shape[1])
    )
    feat_df = feat_df.melt(id_vars=["step"], var_name="type", value_name="value")

    feat_fig = px.line(feat_df, x="step", y="value", color="type")
    feat_fig.update_yaxes(range=(-20, 20))
    feat_fig.update_traces(line_width=1, opacity=0.5)

    import plotly_utility.subplots

    fig.update_traces(legendgroup="prob", legendgrouptitle_text="prob")
    feat_fig.update_traces(legendgroup="feat", legendgrouptitle_text="feat")
    fig = plotly_utility.subplots.vstack(fig, feat_fig)
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

    if cfg.prev_margin_steps + cfg.next_margin_steps > 0:
        interest_start = np.argmax(feat_record["mask"])
        interest_end = interest_start + np.argmin(feat_record["mask"][interest_start:])
        assert np.any(feat_record["mask"][interest_end:]) == np.False_

        fig.add_vrect(x0=interest_start, x1=interest_end)

    fig.update_xaxes(range=(0, cfg.duration))
    fig.update_layout(title=f"{series_id}, chunk_id = {i}")
    fig.show()


plot(0)
plot(1)

plot(22)
plot(23)

plot(43)
plot(44)
plot(45)
plot(46)
