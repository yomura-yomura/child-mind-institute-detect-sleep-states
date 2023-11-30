import pathlib
from typing import Literal, cast

import cmi_dss_lib.datamodule.seg
import cmi_dss_lib.datamodule.stacking
import hydra
import numpy as np
import omegaconf
import pandas as pd
import plotly.express as px
import tqdm
from cmi_dss_lib.config import StackingConfig, TrainConfig
from cmi_dss_lib.datamodule.seg import Indexer

this_dir_path = pathlib.Path(__file__).parent
project_root_path = this_dir_path.parent.parent


# exp_name = "exp041"
# exp_name = "exp050-transformer-decoder_retry_resume"

try:
    hydra.initialize(
        # config_path=str((project_root_path / "run" / "conf").relative_to(this_dir_path)),
        config_path="../conf",
        version_base="1.2",
    )
except ValueError:
    pass


class Plotter:
    def __init__(
        self,
        exp_name: str,
        i_fold: int,
        dataset_type: Literal["train", "valid"],
        inference_step_offset: int = 0,
    ):
        if exp_name.startswith("blending/"):
            overrides_yaml_path = (
                project_root_path
                / "cmi-dss-ensemble-models"
                / "ranchantan/exp050-transformer-decoder_retry_resume"
                / f"fold_{i_fold}"
                / ".hydra"
                / "overrides.yaml"
            )
        else:
            overrides_yaml_path = (
                project_root_path
                / "cmi-dss-ensemble-models"
                / exp_name
                / f"fold_{i_fold}"
                / ".hydra"
                / "overrides.yaml"
            )
        if not overrides_yaml_path.exists():
            overrides_yaml_path = (
                project_root_path
                / "output"
                / "train"
                / exp_name.split("/", maxsplit=1)[1]
                / f"fold_{i_fold}"
                / ".hydra"
                / "overrides.yaml"
            )
        assert overrides_yaml_path.exists(), overrides_yaml_path

        if exp_name.startswith("train_stacking"):
            self.cfg = cast(
                StackingConfig,
                hydra.compose(
                    "stacking", overrides=list(omegaconf.OmegaConf.load(overrides_yaml_path))
                ),
            )
        else:
            self.cfg = cast(
                TrainConfig,
                hydra.compose(
                    "train", overrides=list(omegaconf.OmegaConf.load(overrides_yaml_path))
                ),
            )
        self.cfg.dir.data_dir = (
            project_root_path.parent.parent / "data" / "child-mind-institute-detect-sleep-states"
        )
        self.cfg.dir.sub_dir = project_root_path / "run"

        self.target_pred_dir_path = self.get_pred_dir_path(exp_name, i_fold, inference_step_offset)
        if not self.target_pred_dir_path.exists():
            self.target_pred_dir_path = self.get_pred_dir_path(
                "train/" + exp_name.split("/", maxsplit=1)[1], i_fold, inference_step_offset
            )
        assert self.target_pred_dir_path.exists(), self.target_pred_dir_path

        self.cfg.inference_step_offset = int(inference_step_offset)

        self.events = [
            label[6:] if label.startswith("event_") else label for label in self.cfg.labels
        ]

        # cfg.prev_margin_steps = 6 * 12 * 60
        # cfg.next_margin_steps = 6 * 12 * 60
        if exp_name.startswith("train_stacking"):
            datamodule = cmi_dss_lib.datamodule.stacking.StackingDataModule(self.cfg)
        else:
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

    @staticmethod
    def get_pred_dir_path(exp_name: str, i_fold: int, inference_step_offset: int = 0):
        return (
            project_root_path
            / "run"
            / "predicted"
            / exp_name
            / ("train" if inference_step_offset <= 0 else f"train-cfg.{inference_step_offset=}")
            / f"fold_{i_fold}"
        )

    def get_data(self, chunk_id: int):
        feat_record = self.val_dataset[chunk_id]

        if self.dataset_type == "train":
            series_id = feat_record["series_id"]
            preds = None
            start = end = None
        else:
            series_id, chunk_id = feat_record["key"].split("_")
            chunk_id = int(chunk_id)
            # assert int(feat_record["key"].split("_")[1]) == i

            preds = np.load(self.target_pred_dir_path / f"{series_id}.npz")["arr_0"]
            indexer = Indexer(
                preds.shape[0],
                self.cfg.duration,
                self.cfg.prev_margin_steps,
                self.cfg.next_margin_steps,
                self.cfg.fix_start_timing_hour_with,
                series_id=series_id,
            )
            start, end = indexer.get_cropping_range(chunk_id)
            preds = preds[start:end]
        return (series_id, chunk_id), feat_record, preds, (start, end)

    def get_pred_df(self, i: int):
        _, _, preds, _ = self.get_data(i)
        return self._get_pred_df(preds)

    def _get_pred_df(self, preds):
        if preds is None:
            return None
        pred_df = pd.DataFrame(preds, columns=self.events).assign(step=np.arange(preds.shape[0]))
        pred_df["step"] += self.cfg.inference_step_offset
        pred_df = pred_df.melt(id_vars=["step"], var_name="type", value_name="prob")
        return pred_df

    def get_pred_fig(self, i: int):
        _, _, preds, _ = self.get_data(i)
        return self._get_pred_fig(preds)

    def _get_pred_fig(self, preds):
        pred_df = self._get_pred_df(preds)
        if pred_df is None:
            return None
        fig = px.line(pred_df, x="step", y="prob", color="type")
        fig.update_traces(legendgroup="prob", legendgrouptitle_text="prob")
        return fig

    def get_feat_fig(self, i: int):
        _, feat_record, _, _ = self.get_data(i)
        return self._get_feat_fig(feat_record)

    def _get_feat_fig(self, feat_record):
        feat_df = pd.DataFrame(feat_record["feature"].T, columns=self.cfg.features).assign(
            step=np.arange(feat_record["feature"].shape[1])
        )
        feat_df = feat_df.melt(id_vars=["step"], var_name="type", value_name="value")

        feat_fig = px.line(feat_df, x="step", y="value", color="type")
        feat_fig.update_yaxes(range=(-20, 20))
        feat_fig.update_traces(line_width=1, opacity=0.5)
        feat_fig.update_traces(legendgroup="feat", legendgrouptitle_text="feat")
        return feat_fig

    def get_indices(self, series_id: str | None = None, chunk_id: int | None = None):
        return [
            i
            for i, key in enumerate(self.val_dataset.keys)
            if series_id is None or key.split("_")[0] == series_id
            if chunk_id is None or int(key.split("_")[1]) == chunk_id
        ]

    def plot(self, i: int, do_plot: bool = True):
        (series_id, chunk_id), feat_record, preds, _ = self.get_data(i)

        fig = self._get_pred_fig(preds)

        # fig

        import plotly_utility.subplots

        feat_fig = self._get_feat_fig(feat_record)
        if fig is None:
            fig = feat_fig
        else:
            fig = plotly_utility.subplots.vstack(fig, feat_fig)

        label_df = pd.DataFrame(feat_record["label"], columns=self.events).assign(
            step=2 * np.arange(feat_record["label"].shape[0])
        )

        if "onset" in self.events:
            onset_label_steps = label_df[(label_df["onset"] == 1).astype(bool)]["step"].to_numpy()
            for step in tqdm.tqdm(onset_label_steps, desc="add onset line"):
                fig.add_vline(
                    x=step,
                    annotation_text="onset",
                    line=dict(dash="dash", color="red"),
                )

        if "wakeup" in self.events:
            wakeup_label_steps = label_df[(label_df["wakeup"] == 1).astype(bool)][
                "step"
            ].to_numpy()
            for step in tqdm.tqdm(wakeup_label_steps, desc="add wakeup line"):
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
        fig.update_layout(title=f"{series_id}, {chunk_id=}")
        # fig.update_layout(hovermode="x")

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
        series_id=series_id,
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

    score_dict = {
        event: [metric["average_precision"] for metric in metric_list]
        for event, metric_list in child_mind_institute_detect_sleep_states.score.fast_event_detection_ap.get_score_dict(
            target_event_df, sub_df, show_progress=False
        ).items()
    }
    return score_dict


if __name__ == "__main__":
    # exp_name = "ranchantan/exp050-transformer-decoder_retry_resume"
    # exp_name = "ranchantan/exp104"
    exp_name = "blending/exo024-1"

    plotter50 = Plotter(exp_name, i_fold=0, dataset_type="valid")
    plotter50.plot(5)
    plotter50.plot(0)
    # plotter50.plot(47)

    # get_score(plotter50, 0)

    # plotter58 = Plotter("jumtras/exp058", i_fold=2)
    # plotter58.plot(46)
    # plotter58.plot(47)

    # plotter75 = Plotter("ranchantan/exp075-wakeup_5", i_fold=2, dataset_type="valid")
    # plotter75.plot(5)

    plotter = plotter50

    import child_mind_institute_detect_sleep_states.data.comp_dataset
    import child_mind_institute_detect_sleep_states.score

    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    plotters = [
        Plotter(exp_name, i_fold=i_fold, dataset_type="valid") for i_fold in tqdm.trange(5)
    ]

    records = []
    for i_fold in range(5):
        plotter = plotters[i_fold]
        for p in tqdm.tqdm(sorted(plotter.target_pred_dir_path.glob("*.npz"))):
            series_id = p.stem
            preds = np.load(p)["arr_0"]
            sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
                series_id=series_id,
                preds=preds,
                labels=plotter.cfg.labels,
                downsample_rate=plotter.cfg.downsample_rate,
                score_th=0.0005,
                distance=96,
                post_process_modes=None,
            )
            target_event_df = event_df[event_df["series_id"] == series_id]
            if len(target_event_df) == 0:
                score_onset = score_wakeup = np.nan
            else:
                metric_dict = child_mind_institute_detect_sleep_states.score.fast_event_detection_ap.get_score_dict(
                    target_event_df, sub_df, n_jobs=1
                )
                score_onset = np.mean(
                    [metric["average_precision"] for metric in metric_dict["onset"]]
                )
                score_wakeup = np.mean(
                    [metric["average_precision"] for metric in metric_dict["wakeup"]]
                )

            records.append(
                {
                    "series_id": series_id,
                    "fold": i_fold,
                    "score": np.mean([score_onset, score_wakeup]),
                    "score_onset": score_onset,
                    "score_wakeup": score_wakeup,
                    "n_true_records": len(target_event_df),
                }
            )
    score_df = pd.DataFrame(records)

    #

    series_id = "280e08693c6d"
    target_pred_dir_path = Plotter.get_pred_dir_path(exp_name, 0)
    preds = np.load(target_pred_dir_path / f"{series_id}.npz")["arr_0"]
    sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
        series_id=series_id,
        preds=preds,
        labels=plotter.cfg.labels,
        downsample_rate=plotter.cfg.downsample_rate,
        score_th=0.0005,
        distance=96,
        post_process_modes=None,
    )
    metric_dict = (
        child_mind_institute_detect_sleep_states.score.fast_event_detection_ap.get_score_dict(
            event_df[event_df["series_id"] == series_id].dropna(), sub_df
        )
    )
    metric_df = pd.DataFrame(
        [
            {"event": event, "th": th, "precision": p, "recall": r, "prob": prob}
            for event, metric_list in metric_dict.items()
            for th, metric in enumerate(metric_list)
            for p, r, prob in zip(
                metric["precision"], metric["recall"], metric["prob"], strict=True
            )
        ]
    )
    fig = px.line(
        metric_df, title=series_id, x="recall", y="precision", color="th", facet_col="event"
    )
    fig.update_traces(mode="lines+markers")
    fig.show()

    fig = px.histogram(score_df, x="score", facet_row="fold")
    fig.show()

    fasd

    exp_name = "ranchantan/exp050-transformer-decoder_retry_resume"
    inference_step_offset = 0

    target_pred_dir_path = (
        project_root_path
        / "run"
        / "predicted"
        / exp_name
        / ("train" if inference_step_offset <= 0 else f"train-cfg.{inference_step_offset=}")
    )
    for i_fold in range(5):
        target_pred_fold_dir_path = target_pred_dir_path / f"fold_{i_fold}"

        offset = 360

        records = []
        for i, event in enumerate(["onset", "wakeup"]):
            for series_id, steps in event_df[event_df["event"] == event].groupby("series_id")[
                "step"
            ]:
                try:
                    preds = np.load(target_pred_fold_dir_path / f"{series_id}.npz")["arr_0"]
                except FileNotFoundError:
                    continue

                interest_preds = np.take(
                    preds[:, 1 + i],
                    np.clip(
                        steps.to_numpy("i8")[:, np.newaxis] + np.arange(-offset, offset),
                        0,
                        len(preds) - 1,
                    ),
                )
                for d in interest_preds.argmax(axis=1) - offset:
                    records.append({"series_id": series_id, "event": event, "x": d})
        df = pd.DataFrame(records)
        fig = px.histogram(df, title=f"fold {i_fold + 1}", x="x", color="event", barmode="overlay")
        fig.show()

        import standard_fit as sf

        print(f"fold {i_fold + 1}")
        print(sf.gaussian_fit(df.query("event == 'onset'")["x"], print_result=False)[1])
        print(sf.gaussian_fit(df.query("event == 'wakeup'")["x"], print_result=False)[1])
        print()

    #
