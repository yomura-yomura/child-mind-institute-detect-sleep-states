import argparse
import sys
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import SegDataModule, nearest_valid_size
from cmi_dss_lib.models.common import get_model
from cmi_dss_lib.utils.common import trace
from cmi_dss_lib.utils.post_process import post_process_for_seg
from lightning import seed_everything
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm


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
        project_root_path
        / "output_dataset"
        / cfg.dir.model_dir
        # / cfg.weight["exp_name"]
        # / cfg.weight["run_name"]
        / cfg.exp_name
        / cfg.split.name
        / "best_model.pth"
    )
    model.load_state_dict(torch.load(weight_path))
    print('load weight from "{}"'.format(weight_path))

    return model


# def get_test_dataloader(cfg: DictConfig) -> DataLoader:
#     """get test dataloader
#
#     Args:
#         cfg (DictConfig): config
#
#     Returns:
#         DataLoader: test dataloader
#     """
#     feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
#     series_ids = [x.name for x in feature_dir.glob("*")]
#     chunk_features = load_chunk_features(
#         duration=cfg.duration,
#         feature_names=cfg.features,
#         series_ids=series_ids,
#         processed_dir=Path(cfg.dir.processed_dir),
#         phase=cfg.phase,
#     )
#     test_dataset = TestDataset(cfg, chunk_features=chunk_features)
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=False,
#         num_workers=cfg.num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return test_dataloader


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


def make_submission(keys: list[str], preds: np.ndarray, score_th, distance) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds[:, :, [1, 2]],  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        # test_dataloader = get_test_dataloader(cfg)
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

    if cfg.phase == "train":
        labels = np.concatenate([batch["label"] for batch in dataloader], axis=0)
        np.savez(f"predicted-{cfg.split.name}.npz", key=keys, pred=preds, label=labels)
    else:
        np.savez(f"predicted-{cfg.split.name}.npz", key=keys, pred=preds)

    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            score_th=cfg.post_process.score_th,
            distance=cfg.post_process.distance,
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


import pathlib

import pandas as pd

project_root_path = pathlib.Path(__file__).parent.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args(["config/omura/3090/lstm-feature-extractor.yaml"])

    for i_fold in range(5):
        overrides_args = OmegaConf.load(project_root_path / args.config_path)
        overrides_args.append(f"split=fold_{i_fold}")
        sys.argv = sys.argv[:1] + overrides_args
        main()
