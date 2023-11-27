import argparse
import os
import pathlib

import cmi_dss_lib.utils.common
import cmi_dss_lib.utils.hydra
import cmi_dss_lib.utils.inference
import hydra
import lightning as L
import numpy as np
import torch
from cmi_dss_lib.config import StackingConfig
from cmi_dss_lib.datamodule.stacking import StackingDataModule
from cmi_dss_lib.modelmodule.stacking import StackingChunkModule
from cmi_dss_lib.utils.common import trace

import child_mind_institute_detect_sleep_states.pj_struct_paths

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        # "../../output/train_stacking/s_exp006",
        # "../../cmi-dss-ensemble-stacking-models/s_exp006",
        "../../output/train_stacking/s_exp006",
        #
        # "phase=dev",
        # "phase=train",
        "phase=test",
        # "batch_size=32",
        "batch_size=16",
    ]
else:
    args = None

project_root_path = pathlib.Path(__file__).parent.parent.parent


def load_model(cfg: StackingConfig) -> L.LightningModule:
    model_fold_dir_path = pathlib.Path(cfg.dir.model_dir)

    if (weight_path := model_fold_dir_path / "best.ckpt").exists():
        module = StackingChunkModule.load_from_checkpoint(
            weight_path,
            cfg=cfg,
        )
    else:
        module = StackingChunkModule(cfg)

        weight_path = model_fold_dir_path / "best_model.pth"
        module.model.load_state_dict(torch.load(weight_path))
    print(f'load weight from "{weight_path}"')

    return module


@hydra.main(config_path="../conf", config_name="stacking", version_base="1.2")
def main(cfg: StackingConfig):
    print(cfg)
    child_mind_institute_detect_sleep_states.pj_struct_paths.set_pj_struct_paths(
        kaggle_dataset_dir_path=cfg.dir.data_dir
    )

    with trace("load model"):
        module = load_model(cfg)

    datamodule = StackingDataModule(cfg)

    score = cmi_dss_lib.utils.inference.run(cfg, module, datamodule)
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
        main()
        cmi_dss_lib.utils.common.clean_memory()

    mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
    print(f"{mean_score_str} ({', '.join(score_strs)})")
