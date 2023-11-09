import argparse
import os
import pathlib
import sys

import cmi_dss_lib.utils.metrics
import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import SegDataModule, nearest_valid_size
from cmi_dss_lib.models.common import get_model
from cmi_dss_lib.utils.common import trace
from cmi_dss_lib.utils.post_process import PostProcessModes, post_process_for_seg
from lightning import seed_everything
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm

project_root_path = pathlib.Path(__file__).parent.parent


if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        # "../cmi-dss-ensemble-models/jumtras/exp016-gru-feature-fp16-layer4-ep70-lr-half",
        # "../cmi-dss-ensemble-models/ranchantan/exp005-lstm-feature-2",
        # "../output_dataset/train/exp016-1d-resnet34"
        "../output_dataset/train/exp015-lstm-feature-108-sigma"
    ]
else:
    args = None


def load_model(cfg: TrainConfig) -> nn.Module:
    num_time_steps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_time_steps=num_time_steps // cfg.downsample_rate,
    )

    # load weights
    weight_path = (
        # project_root_path
        # / "output_dataset"
        pathlib.Path(cfg.dir.model_dir)
        # / cfg.weight["exp_name"]
        # / cfg.weight["run_name"]
        # / cfg.exp_name
        # / cfg.split.name
        / "best_model.pth"
    )
    model.load_state_dict(torch.load(weight_path))
    print(f'load weight from "{weight_path}"')

    return model


def inference(
    duration: int, loader: DataLoader, model: nn.Module, device: torch.device, use_amp
) -> tuple[list[str], np.ndarray]:
    model = model.to(device)
    model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                pred = model(x)["logits"].sigmoid()
                pred = resize(
                    pred.detach().cpu(),
                    size=[duration, pred.shape[2]],
                    antialias=False,
                )
            key = batch["key"]
            preds.append(pred.detach().cpu().numpy())
            keys.extend(key)

    preds = np.concatenate(preds)

    return keys, preds  # type: ignore


def make_submission(
    keys: list[str],
    preds: np.ndarray,
    downsample_rate: int,
    score_th: float,
    distance: int,
    post_process_modes: PostProcessModes = None,
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds,
        downsample_rate=downsample_rate,
        score_th=score_th,
        distance=distance,
        post_process_modes=post_process_modes,
    )

    return sub_df


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        data_module = SegDataModule(cfg)

        if cfg.phase == "train":
            data_module.setup("valid")
            dataloader = data_module.val_dataloader()
        elif cfg.phase == "test":
            data_module.setup("test")
            dataloader = data_module.test_dataloader()

    with trace("load model"):
        model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        keys, preds = inference(cfg.duration, dataloader, model, device, use_amp=cfg.use_amp)

    pred_dir_path = pathlib.Path(
        cfg.dir.sub_dir, "predicted", *pathlib.Path(cfg.dir.model_dir).parts[-3:-1]
    )
    pred_dir_path.mkdir(parents=True, exist_ok=True)
    if cfg.phase == "train":
        labels = np.concatenate([batch["label"] for batch in dataloader], axis=0)
        np.savez(
            pred_dir_path / f"predicted-{cfg.split.name}.npz",
            key=keys,
            pred=preds,
            label=labels,
        )
    else:
        np.savez(
            pred_dir_path / f"predicted-{cfg.split.name}.npz",
            key=keys,
            pred=preds,
        )

    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            downsample_rate=cfg.downsample_rate,
            score_th=cfg.post_process.score_th,
            distance=cfg.post_process.distance,
        )

    if cfg.phase == "train":
        unique_series_ids = np.unique([str(k).split("_")[0] for k in keys])

        event_df = pd.read_csv(pathlib.Path(cfg.dir.data_dir) / "train_events.csv")
        event_df = event_df[event_df["series_id"].isin(unique_series_ids)].dropna()

        score = cmi_dss_lib.utils.metrics.event_detection_ap(event_df, sub_df.to_pandas())
        print(f"{cfg.split.name}: {score:.4f}")

    sub_df.write_csv(pathlib.Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("config_path", nargs="*")
    args = parser.parse_args(args)

    for i_fold in range(5):
        overrides_args = []

        fold_dir_path = args.model_path / f"fold_{i_fold}"
        if not fold_dir_path.exists():
            raise FileNotFoundError(fold_dir_path)

        for p in (
            fold_dir_path / ".hydra" / "overrides.yaml",
            *args.config_path,
        ):
            overrides_args += OmegaConf.load(p)
        overrides_args.append(f"split=fold_{i_fold}")
        overrides_args.append(f"dir.model_dir={args.model_path / f'fold_{i_fold}'}")
        # overrides_args.append(f"phase=test")
        sys.argv = sys.argv[:1] + overrides_args
        main()
