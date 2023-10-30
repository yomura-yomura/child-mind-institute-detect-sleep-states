import json
import pathlib
import shutil
import tempfile

base_path = pathlib.Path("models/multi_res_bi_gru/remove-0.8-nan")

dataset_dir_path = pathlib.Path(tempfile.mkdtemp())

shutil.copy(base_path / "config.toml", dataset_dir_path / "config.toml")

for p in base_path.glob("fold*"):
    src_path = (p / "best.ckpt").readlink()
    dataset_fold_dir_path = dataset_dir_path / p.name
    dataset_fold_dir_path.mkdir(exist_ok=True)
    shutil.copy(src_path, dataset_fold_dir_path / "best.ckpt")
    shutil.copy(p / "prob.csv", dataset_fold_dir_path / "prob.csv")


dataset_metadata_json = {
    "title": "CMI-DSS MultiResBiGru",
    "id": "ranchantan/cmi-dss-multi-res-bi-gru",
    "licenses": [{"name": "CC0-1.0"}],
}

import subprocess

with open(dataset_dir_path / "dataset-metadata.json", "w") as f:
    json.dump(dataset_metadata_json, f, indent=2)


subprocess.run(
    " ".join(
        [
            "/home/yugo.omura/.local/bin/kaggle",
            "datasets",
            "version",
            "-p",
            str(dataset_dir_path),
            "-m",
            "''",
            "--dir-mode",
            "tar",
        ]
    ),
    shell=True,
)
