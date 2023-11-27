import json
import pathlib
import shutil
import subprocess
import sys

import hydra
import pandas as pd
import torch
import tqdm
from cmi_dss_lib.modelmodule.seg import SegChunkModule
from cmi_dss_lib.modelmodule.stacking import StackingChunkModule
from omegaconf import DictConfig, OmegaConf

project_root_path = pathlib.Path(__file__).parent.parent

# train_type = "train"
# exp_name = "exp005-lstm-feature"
# exp_name = "exp005-lstm-feature-2"
# exp_name = "exp014-lstm-feature"
# exp_name = "exp015-lstm-feature-sigma"

# exp_name = "exp008-lstm-feature-half-lr"
# exp_name = "exp012-lstm-feature"t
# exp_name = "exp011-lstm-feature-1d-fp16"

# exp_name = "exp016-1d-resnet34"
# exp_name = "exp015-lstm-feature-108-sigma"
# exp_name = "exp008-3sigma"
# exp_name = "exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma"
# exp_name = "exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation"
# exp_name = "exp045-lstm-feature-extractor"
# exp_name = "exp044-transformer-decoder"

# exp_name = "exp034-stacked-gru-4-layers-24h-duration-4bs-108sigma-no-bg_sampling_rate"
# exp_name = "exp041_retry"
# exp_name = "exp047_retry"
# exp_name = "exp050-transformer-decoder"
# exp_name = "exp050-transformer-decoder_retry"
# exp_name = "exp050-transformer-decoder_retry_resume"
# exp_name = "exp054"
# exp_name = "exp055"
# exp_name = "exp060"
# exp_name = "exp073_resume"
# exp_name = "exp075-onset"

train_type = "stacking"
exp_name = "s_exp006"

# upload = False
upload = True


@hydra.main(config_path="conf", config_name=train_type, version_base="1.2")
def main(cfg: DictConfig):
    if train_type == "train":
        dataset_output_dir_path = project_root_path / "cmi-dss-ensemble-models" / "ranchantan"
        module = SegChunkModule.load_from_checkpoint(
            f"{p}",
            cfg=cfg,
            val_event_df=None,
            feature_dim=len(cfg.features),
            num_classes=len(cfg.labels),
            duration=cfg.duration,
            map_location="cpu",
        )
    elif train_type == "stacking":
        dataset_output_dir_path = project_root_path / "cmi-dss-ensemble-stacking-models"
        module = StackingChunkModule.load_from_checkpoint(
            f"{p}",
            cfg=cfg,
        )
    else:
        raise ValueError(f"unexpected {train_type=}")

    output_dir_path = dataset_output_dir_path / p.relative_to(train_output_dir_path).parent
    output_dir_path.mkdir(exist_ok=True, parents=True)

    (output_dir_path / ".hydra").mkdir(exist_ok=True)
    shutil.copy(
        p.parent / ".hydra" / "overrides.yaml", output_dir_path / ".hydra" / "overrides.yaml"
    )

    torch.save(
        module.model.state_dict(),
        output_dir_path / "best_model.pth",
    )


if __name__ == "__main__":
    train_output_dir_path = (
        project_root_path / "output" / ("train" if train_type == "train" else "train_stacking")
    )
    print(f"{train_output_dir_path.resolve()=}")

    path_df = pd.DataFrame(
        [
            (
                p,
                int(p.parts[-2][5:]),
                int(p.parts[-1][6:-5]) if "-v" in p.parts[-1] else 0,
            )
            for p in train_output_dir_path.glob(f"{exp_name}/fold_*/best*.ckpt")
        ],
        columns=["path", "i_fold", "version"],
    )
    path_df = (
        path_df.sort_values(["i_fold", "version"], ascending=[True, False])
        .groupby(["i_fold"])
        .head(1)
    )
    print(path_df)

    scores = []
    for p, i_fold, version in tqdm.tqdm(path_df.itertuples(index=False)):
        print(p.readlink().name)
        scores.append(float(p.readlink().stem.split("EventDetectionAP=", maxsplit=1)[1]))
        p = p.parent / p.readlink().name
        overrides_args = OmegaConf.load(p.parent / ".hydra" / "overrides.yaml")
        sys.argv = overrides_args
        main()

    print(f"CV = {sum(scores) / len(scores):.3f} ({', '.join(map('{:.3f}'.format, scores))})")

    if upload:
        if train_type == "train":
            dataset_metadata_json = {
                "title": "CMI-DSS Segmentation Model",
                "id": "ranchantan/cmi-dss-seg-model",
                "licenses": [{"name": "CC0-1.0"}],
            }
        else:
            dataset_metadata_json = {
                "title": "CMI-DSS Segmentation Stacking Model",
                "id": "ranchantan/cmi-dss-seg-stacking-model",
                "licenses": [{"name": "CC0-1.0"}],
            }

        dataset_dir_path = project_root_path / (
            "cmi-dss-ensemble-models"
            if train_type == "train"
            else "cmi-dss-ensemble-stacking-models"
        )
        dataset_dir_path.mkdir(exist_ok=True, parents=True)
        with open(dataset_dir_path / "dataset-metadata.json", "w") as f:
            json.dump(dataset_metadata_json, f, indent=2)

        (dataset_dir_path / ".keep").touch(exist_ok=True)

        subprocess.run(
            " ".join(
                [
                    str(pathlib.Path.home() / ".local" / "bin" / "kaggle"),
                    "datasets",
                    "version",
                    "-p",
                    f"{dataset_dir_path}",
                    "-m",
                    "''",
                    "--dir-mode",
                    "tar",
                ]
            ),
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
        )
