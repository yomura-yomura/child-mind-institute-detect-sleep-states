import argparse
import os
import pathlib

import cmi_dss_lib.utils.hydra
import hydra
from cmi_dss_lib.datamodule.stacking import StackingConfig, StackingDataModule
from cmi_dss_lib.modelmodule.stacking import StackingChunkModule
from lightning import seed_everything

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        "../config/exp_for_stacking/1.yaml",
        # "--folds", "1,2,3,4"
    ]
else:
    args = None


project_root_path = pathlib.Path(__file__).parent.parent


@hydra.main(config_path="conf", config_name="stacking", version_base="1.2")
def main(cfg: StackingConfig):
    print(cfg)

    seed_everything(cfg.seed)

    datamodule = StackingDataModule(cfg)
    datamodule.setup("fit")
    x = next(iter(datamodule.train_dataloader()))

    model_save_dir_path = (
        project_root_path / cfg.dir.output_dir / "train_stacking" / cfg.exp_name / cfg.split.name
    )

    module = StackingChunkModule(
        cfg, val_event_df=datamodule.valid_event_df, model_save_dir_path=model_save_dir_path
    )
    preds = module(x)
    print(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path_or_hydra_arguments", nargs="+")
    parser.add_argument("--folds", type=str, default=None)
    args = parser.parse_args(args)

    if args.folds is None:
        folds = list(range(5))
    else:
        folds = list(map(int, args.folds.split(",")))

    print(f"{folds = }")

    for i_fold in folds:
        cmi_dss_lib.utils.hydra.override_default_hydra_config(
            args.config_path_or_hydra_arguments, {"split": f"fold_{i_fold}"}
        )
        main()
        break
