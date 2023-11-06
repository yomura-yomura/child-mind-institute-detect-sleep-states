import json
import pathlib
import subprocess
import sys

import hydra
import pandas as pd
import torch
import tqdm
from cmi_dss_lib.modelmodule.seg import SegModel
from omegaconf import DictConfig

project_root_path = pathlib.Path(__file__).parent.parent


exp_name = "exp005-lstm-feature-2"
# exp_name = "exp008-lstm-feature-half-lr"
sys.argv += ["feature_extractor=LSTMFeatureExtractor"]


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    train_output_dir_path = project_root_path / "output"
    dataset_output_dir_path = project_root_path / "output_dataset"
    print(f"{train_output_dir_path.resolve()=}")

    path_df = pd.DataFrame(
        [
            (
                p,
                int(p.parts[-3].split("-fold", maxsplit=1)[1]),
                int(p.parts[-1][6:-5]) if "-v" in p.parts[-1] else 0,
            )
            for p in train_output_dir_path.glob(f"train/{exp_name}-fold*/single/best*.ckpt")
        ],
        columns=["path", "i_fold", "version"],
    )
    path_df = (
        path_df.sort_values(["i_fold", "version"], ascending=[True, False])
        .groupby(["i_fold"])
        .head(1)
    )

    scores = []
    for p, i_fold, version in tqdm.tqdm(path_df.itertuples(index=False)):
        print(p.readlink().name)
        scores.append(float(p.readlink().stem.split("EventDetectionAP=", maxsplit=1)[1]))
        module = SegModel.load_from_checkpoint(
            f"{p}",
            cfg=cfg,
            val_event_df=None,
            feature_dim=len(cfg.features),
            num_classes=len(cfg.labels),
            duration=cfg.duration,
            map_location="cpu",
        )

        output_dir_path = dataset_output_dir_path / p.relative_to(train_output_dir_path).parent
        output_dir_path.mkdir(exist_ok=True, parents=True)
        torch.save(
            module.model.state_dict(),
            output_dir_path / "best_model.pth",
        )

    print(f"CV = {sum(scores) / len(scores):.3f} ({', '.join(map('{:.3f}'.format, scores))})")

    dataset_metadata_json = {
        "title": "CMI-DSS Segmentation Model",
        "id": "ranchantan/cmi-dss-seg-model",
        "licenses": [{"name": "CC0-1.0"}],
    }

    dataset_dir_path = project_root_path / "output_dataset" / "train"
    dataset_dir_path.mkdir(exist_ok=True, parents=True)
    with open(dataset_dir_path / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata_json, f, indent=2)

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


if __name__ == "__main__":
    main()
