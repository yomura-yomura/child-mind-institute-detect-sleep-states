import pathlib
from typing import Literal, cast

import cmi_dss_lib.datamodule.seg
import hydra
import numpy as np
import omegaconf
import pandas as pd
import plotly.express as px
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import Indexer

project_root_path = pathlib.Path(__file__).parent.parent.parent


# exp_name = "exp041"
# exp_name = "exp050-transformer-decoder_retry_resume"


hydra.initialize(config_path="../conf", version_base="1.2")


class Plotter:
    def __init__(self, exp_name: str, i_fold: int, dataset_type: Literal["train", "valid"]):
        self.target_pred_dir_path = project_root_path / "run" / "predicted" / exp_name / "train"
        assert self.target_pred_dir_path.exists()

        overrides_yaml_path = (
            project_root_path
            / "cmi-dss-ensemble-models"
            / exp_name
            / f"fold_{i_fold}"
            / ".hydra"
            / "overrides.yaml"
        )
        assert overrides_yaml_path.exists()

        self.cfg = cast(
            TrainConfig,
            hydra.compose("train", overrides=list(omegaconf.OmegaConf.load(overrides_yaml_path))),
        )
        self.cfg.dir.data_dir = (
            project_root_path.parent.parent / "data" / "child-mind-institute-detect-sleep-states"
        )

        # cfg.prev_margin_steps = 6 * 12 * 60
        # cfg.next_margin_steps = 6 * 12 * 60

        datamodule = cmi_dss_lib.datamodule.seg.SegDataModule(self.cfg)

        self.dataset_type = dataset_type
        if dataset_type == "train":
            datamodule.setup("fit")
            self.val_dataset = datamodule.train_dataloader().dataset
        elif dataset_type == "valid":
            datamodule.setup("valid")
            self.val_dataset = datamodule.val_dataloader().dataset
        else:
            raise ValueError(f"unexpected {dataset_type=}")

    def get_data(self, i: int):
        feat_record = self.val_dataset[i]

        if self.dataset_type == "train":
            series_id = feat_record["series_id"]
            preds = None
            start = end = None
        else:
            series_id, i = feat_record["key"].split("_")
            i = int(i)
            # assert int(feat_record["key"].split("_")[1]) == i

            preds = np.load(self.target_pred_dir_path / self.cfg.split.name / f"{series_id}.npz")[
                "arr_0"
            ]
            indexer = Indexer(
                preds.shape[0],
                self.cfg.duration,
                self.cfg.prev_margin_steps,
                self.cfg.next_margin_steps,
            )
            start, end = indexer.get_cropping_range(i)
            preds = preds[start:end]
        return (series_id, i), feat_record, preds, (start, end)

    def plot(self, i: int, do_plot: bool = True):
        (series_id, i), feat_record, preds, _ = self.get_data(i)

        events = [label[6:] if label.startswith("event_") else label for label in self.cfg.labels]

        fig = None
        if preds is not None:
            pred_df = pd.DataFrame(preds, columns=events).assign(step=np.arange(preds.shape[0]))
            pred_df = pred_df.melt(id_vars=["step"], var_name="type", value_name="prob")

            fig = px.line(pred_df, x="step", y="prob", color="type")
            fig.update_traces(legendgroup="prob", legendgrouptitle_text="prob")

        label_df = pd.DataFrame(feat_record["label"], columns=events).assign(
            step=2 * np.arange(feat_record["label"].shape[0])
        )

        # fig

        feat_df = pd.DataFrame(feat_record["feature"].T, columns=self.cfg.features).assign(
            step=np.arange(feat_record["feature"].shape[1])
        )
        feat_df = feat_df.melt(id_vars=["step"], var_name="type", value_name="value")

        feat_fig = px.line(feat_df, x="step", y="value", color="type")
        feat_fig.update_yaxes(range=(-20, 20))
        feat_fig.update_traces(line_width=1, opacity=0.5)

        import plotly_utility.subplots

        feat_fig.update_traces(legendgroup="feat", legendgrouptitle_text="feat")
        if fig is None:
            fig = feat_fig
        else:
            fig = plotly_utility.subplots.vstack(fig, feat_fig)

        if "onset" in events:
            onset_label_steps = label_df[label_df["onset"].astype(bool)]["step"].to_numpy()
            for step in onset_label_steps:
                fig.add_vline(
                    x=step,
                    annotation_text="onset",
                    line=dict(dash="dash", color="red"),
                )

        if "wakeup" in events:
            wakeup_label_steps = label_df[label_df["wakeup"].astype(bool)]["step"].to_numpy()
            for step in wakeup_label_steps:
                fig.add_vline(
                    x=step,
                    annotation_text="wakeup",
                    line=dict(dash="dash", color="green"),
                )

        if self.cfg.prev_margin_steps + self.cfg.next_margin_steps > 0:
            interest_start = np.argmax(feat_record["mask"])
            interest_end = interest_start + np.argmin(feat_record["mask"][interest_start:])
            assert np.any(feat_record["mask"][interest_end:]) == np.False_

            fig.add_vrect(x0=interest_start, x1=interest_end)

        fig.update_xaxes(range=(0, self.cfg.duration), matches="x", exponentformat="none")
        fig.update_layout(title=f"{series_id}, chunk_id = {i}")
        fig.update_layout(hovermode="x")

        if do_plot:
            fig.show()
        return fig


import cmi_dss_lib.utils.post_process
from nptyping import DataFrame, NDArray

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.score

event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")


def get_sub_df(
    series_id: str, preds: NDArray, labels: list[str], score_th=0.0005, distance=96
) -> DataFrame:
    sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
        keys=[series_id] * len(preds),
        preds=preds,
        labels=labels,
        downsample_rate=2,
        score_th=score_th,
        distance=distance,
    )
    return sub_df


def get_score(
    series_id: str, sub_df: DataFrame, labels: list[str], start: int, end: int
) -> dict[str, list[float]]:
    target_event_df = event_df[
        (event_df["series_id"] == series_id)
        & (start <= event_df["step"])
        & (event_df["step"] <= end)
        & (event_df["event"].isin(labels))
    ].copy()
    target_event_df["step"] -= start

    if len(target_event_df) == 0:
        return {}

    score_dict = (
        child_mind_institute_detect_sleep_states.score.fast_event_detection_ap.get_score_dict(
            target_event_df, sub_df, show_progress=False
        )
    )
    return score_dict


if __name__ == "__main__":
    plotter50 = Plotter(
        "ranchantan/exp050-transformer-decoder_retry_resume", i_fold=2, dataset_type="valid"
    )
    plotter50.plot(5)
    # plotter50.plot(47)

    # get_score(plotter50, 0)

    # plotter58 = Plotter("jumtras/exp058", i_fold=2)
    # plotter58.plot(46)
    # plotter58.plot(47)

    plotter75 = Plotter("ranchantan/exp075-wakeup_5", i_fold=2, dataset_type="valid")
    plotter75.plot(5)
