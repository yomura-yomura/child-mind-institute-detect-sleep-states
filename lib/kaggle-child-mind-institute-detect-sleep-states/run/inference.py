import argparse
import os
import pathlib
import sys

import cmi_dss_lib.utils.common
import hydra
import lightning as L
import numpy as np
import torch
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import SegDataModule
from cmi_dss_lib.modelmodule.seg import SegChunkModule
from cmi_dss_lib.utils.common import trace
from cmi_dss_lib.utils.post_process import PostProcessModes, SubmissionDataFrame, post_process_for_seg
from lightning import seed_everything
from nptyping import Float, NDArray, Shape
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

project_root_path = pathlib.Path(__file__).parent.parent


if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        "../cmi-dss-ensemble-models/jumtras/exp016-gru-feature-fp16-layer4-ep70-lr-half",  # 3
        # "../cmi-dss-ensemble-models/ranchantan/exp005-lstm-feature-2",
        # "../cmi-dss-ensemble-models/ranchantan/exp016-1d-resnet34",  # 1
        # "../cmi-dss-ensemble-models/ranchantan/exp015-lstm-feature-108-sigma",
        # "../cmi-dss-ensemble-models/ranchantan/exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma/",
        # "../cmi-dss-ensemble-models/jumtras/exp027-TimesNetFeatureExtractor-1DUnet-Unet/",
        # "../cmi-dss-ensemble-models/ranchantan/exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation",
        # "../cmi-dss-ensemble-models/ranchantan/exp050-transformer-decoder",
        # "../cmi-dss-ensemble-models/jumtras/exp043",
        # "../cmi-dss-ensemble-models/ranchantan/exp045-lstm-feature-extractor",
        # "../cmi-dss-ensemble-models/ranchantan/exp044-transformer-decoder",
        #
        # "../cmi-dss-ensemble-models/ranchantan/exp041_retry",
        # "../cmi-dss-ensemble-models/ranchantan/exp047_retry",
        # "../cmi-dss-ensemble-models/ranchantan/exp050-transformer-decoder_retry",
        # "../cmi-dss-ensemble-models/ranchantan/exp050-transformer-decoder_retry_resume",
        # "../cmi-dss-ensemble-models/jumtras/exp052",
        # "../cmi-dss-ensemble-models/jumtras/exp053",
        #
        "phase=dev",
        # "phase=train",
        # "batch_size=32",
        # "batch_size=16",
        "batch_size=8",
        # "--folds",
        # "3,4",
        #
        # "dir.sub_dir=tmp",
        # "prev_margin_steps=4320",
        # "next_margin_steps=4320",
    ]
else:
    args = None


def load_model(cfg: TrainConfig) -> L.LightningModule:
    module = SegChunkModule(
        cfg,
        val_event_df=None,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )

    # load weights
    weight_path = pathlib.Path(cfg.dir.model_dir) / "best_model.pth"
    module.model.load_state_dict(torch.load(weight_path))
    print(f'load weight from "{weight_path}"')

    return module


def inference(loader: DataLoader, model: L.LightningModule, use_amp: bool, pred_dir_path: pathlib.Path):
    trainer = L.Trainer(
        devices=1,
        precision=16 if use_amp else 32,
    )
    predictions = trainer.predict(model, loader)

    series_id_preds_dict = dict(SegChunkModule._evaluation_epoch_end([pred for preds in predictions for pred in preds]))

    for series_id, preds in series_id_preds_dict.items():
        np.savez_compressed(pred_dir_path / f"{series_id}.npz", preds.astype("f2"))


def make_submission(
    keys: list[str],
    preds: NDArray[Shape["*, 3"], Float],
    downsample_rate: int,
    score_th: float,
    distance: int,
    post_process_modes: PostProcessModes = None,
) -> SubmissionDataFrame:
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
    print(cfg)
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        data_module = SegDataModule(cfg)

        if cfg.phase == "train":
            data_module.setup("valid")
            dataloader = data_module.val_dataloader()
        elif cfg.phase == "test":
            data_module.setup("test")
            dataloader = data_module.test_dataloader()
        elif cfg.phase == "dev":
            data_module.setup("dev")
            dataloader = data_module.test_dataloader()
        else:
            raise ValueError(f"unexpected {cfg.phase=}")

    with trace("load model"):
        model = load_model(cfg)

    pred_dir_path = pathlib.Path(
        cfg.dir.sub_dir,
        "predicted",
        *pathlib.Path(cfg.dir.model_dir).parts[-3:-1],
        cfg.phase,
        f"{cfg.split.name}",
    )

    pred_dir_path.mkdir(exist_ok=True, parents=True)
    with trace("inference"):
        # keys, preds = inference_(dataloader, model, use_amp=cfg.use_amp, duration=cfg.duration)
        inference(dataloader, model, use_amp=cfg.use_amp, pred_dir_path=pred_dir_path)

    if cfg.phase == "train":
        from calc_cv import calc_score

        score = calc_score(pred_dir_path)
        print(f"{score:.4f}")
        scores.append(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("config_path_or_hydra_arguments", nargs="*")
    parser.add_argument("--folds", type=str, default=None)
    args = parser.parse_args(args)

    if args.folds is None:
        folds = list(range(5))
    else:
        folds = list(map(int, args.folds.split(",")))

    print(f"{folds = }")

    scores = []
    for i_fold in folds:
        overrides_dict = {}

        fold_dir_path = args.model_path / f"fold_{i_fold}"
        if not fold_dir_path.exists():
            raise FileNotFoundError(fold_dir_path)

        for p in (
            fold_dir_path / ".hydra" / "overrides.yaml",
            *args.config_path_or_hydra_arguments,
        ):
            if os.path.exists(p):
                for k, v in (item.split("=", maxsplit=1) for item in OmegaConf.load(p)):
                    if k in overrides_dict.keys():
                        print(f"Info: {k}={overrides_dict[k]} is replaced with {k}={v}")
                    overrides_dict[k] = v
            else:
                k, v = p.split("=", maxsplit=1)
                if k in overrides_dict.keys():
                    print(f"Info: {k}={overrides_dict[k]} is replaced with {k}={v}")
                overrides_dict[k] = v
        overrides_dict["split"] = f"fold_{i_fold}"
        overrides_dict["dir.model_dir"] = f"{args.model_path / f'fold_{i_fold}'}"
        sys.argv = sys.argv[:1] + [f"{k}={v}" for k, v in overrides_dict.items()]
        main()
        cmi_dss_lib.utils.common.clean_memory()

    mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
    print(f"{mean_score_str} ({', '.join(score_strs)})")
