import argparse
import os
import pathlib
import sys

import cmi_dss_lib.utils.common
import cmi_dss_lib.utils.hydra
import cmi_dss_lib.utils.inference
import hydra
import lightning as L
import numpy as np
import torch
from cmi_dss_lib.config import TrainConfig
from cmi_dss_lib.datamodule.seg import SegDataModule
from cmi_dss_lib.modelmodule.seg import SegChunkModule
from cmi_dss_lib.utils.common import trace

import child_mind_institute_detect_sleep_states.pj_struct_paths

project_root_path = pathlib.Path(__file__).parent.parent


if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        # "../cmi-dss-ensemble-models/jumtras/exp016-gru-feature-fp16-layer4-ep70-lr-half",  # 3
        # "../cmi-dss-ensemble-models/ranchantan/exp005-lstm-feature-2",
        # "../cmi-dss-ensemble-models/ranchantan/exp016-1d-resnet34",  # 1
        # "../cmi-dss-ensemble-models/ranchantan/exp015-lstm-feature-108-sigma",
        # "../cmi-dss-ensemble-models/ranchantan/exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation",
        # "../cmi-dss-ensemble-models/ranchantan/exp050-transformer-decoder",
        # "../cmi-dss-ensemble-models/jumtras/exp043",
        # "../cmi-dss-ensemble-models/ranchantan/exp045-lstm-feature-extractor",
        # "../cmi-dss-ensemble-models/ranchantan/exp044-transformer-decoder",
        #
        # "../cmi-dss-ensemble-models/ranchantan/exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma/",
        # "../cmi-dss-ensemble-models/jumtras/exp027-TimesNetFeatureExtractor-1DUnet-Unet/",
        # "../cmi-dss-ensemble-models/ranchantan/exp034-stacked-gru-4-layers-24h-duration-4bs-108sigma-no-bg_sampling_rate",
        # "../cmi-dss-ensemble-models/ranchantan/exp041_retry",
        # "../cmi-dss-ensemble-models/ranchantan/exp047_retry",
        # "../cmi-dss-ensemble-models/ranchantan/exp050-transformer-decoder_retry",
        # "../cmi-dss-ensemble-models/ranchantan/exp050-transformer-decoder_retry_resume",
        # "../cmi-dss-ensemble-models/jumtras/exp052",
        # "../cmi-dss-ensemble-models/jumtras/exp053",
        # "../cmi-dss-ensemble-models/ranchantan/exp054",
        # "../cmi-dss-ensemble-models/ranchantan/exp055",
        # "../cmi-dss-ensemble-models/jumtras/exp058",
        # "../cmi-dss-ensemble-models/jumtras/exp085",
        # "../cmi-dss-ensemble-models/ranchantan/exp060",
        # "../cmi-dss-ensemble-models/ranchantan/exp073_resume",
        # "../cmi-dss-ensemble-models/ranchantan/exp075-onset",
        # "../cmi-dss-ensemble-models/ranchantan/exp075-wakeup_6",
        # "../output/train/exp095",
        "../cmi-dss-ensemble-models/ranchantan/exp100",
        # "../output/train/exp104_2",
        #
        # "phase=dev",
        "phase=train",
        "batch_size=32",
        # "batch_size=16",
        # "batch_size=8",
        # "--folds",
        # "2,3,4",
        # "inference_step_offset=0,2880,5760,8640,11520,14400",
        #
        # "dir.sub_dir=tmp",
        # "prev_margin_steps=4320",
        # "next_margin_steps=4320",
        # "--multirun",
    ]
else:
    args = None


def load_model(cfg: TrainConfig) -> L.LightningModule:
    model_fold_dir_path = pathlib.Path(cfg.dir.model_dir)

    if (weight_path := model_fold_dir_path / "best.ckpt").exists():
        module = SegChunkModule.load_from_checkpoint(
            weight_path,
            cfg=cfg,
            val_event_df=None,
            feature_dim=len(cfg.features),
            num_classes=len(cfg.labels),
            duration=cfg.duration,
        )
    else:
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


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    print(cfg)

    child_mind_institute_detect_sleep_states.pj_struct_paths.set_pj_struct_paths(
        kaggle_dataset_dir_path=cfg.dir.data_dir
    )

    with trace("load model"):
        module = load_model(cfg)

    datamodule = SegDataModule(cfg)

    score = cmi_dss_lib.utils.inference.run(cfg, module, datamodule)
    if score is not None:
        scores.append(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("config_path_or_hydra_arguments", nargs="*")
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--multirun", action="store_true", default=False)
    args = parser.parse_args(args)

    if args.folds is None:
        folds = list(range(5))
    else:
        folds = list(map(int, args.folds.split(",")))

    print(f"{folds = }")

    scores = []
    for i_fold in folds:
        fold_dir_path = args.model_path / f"fold_{i_fold}"
        if not fold_dir_path.exists():
            raise FileNotFoundError(fold_dir_path)

        cmi_dss_lib.utils.hydra.override_default_hydra_config(
            [
                fold_dir_path / ".hydra" / "overrides.yaml",
                *args.config_path_or_hydra_arguments,
            ],
            overrides_dict={
                "split": f"fold_{i_fold}",
                "dir.model_dir": f"{args.model_path / f'fold_{i_fold}'}",
            },
        )
        if args.multirun:
            sys.argv.insert(
                1,
                "--multirun",
            )

        main()
        cmi_dss_lib.utils.common.clean_memory()

    mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
    print(f"{mean_score_str} ({', '.join(score_strs)})")
