import json
import pathlib
import subprocess
import sys

subprocess.run(["rm", "-rf", "child-mind-institute-detect-sleep-states"])

subprocess.run(["git", "clone", "git@github.com:yomura-yomura/child-mind-institute-detect-sleep-states.git"])


dataset_metadata_json = {
    "title": "CMI-DSS SegModel Repo",
    "id": "ranchantan/cmi-dss-seg-model-repo",
    "licenses": [{"name": "CC0-1.0"}],
}

dataset_dir_path = (
    pathlib.Path("child-mind-institute-detect-sleep-states") / "lib" / "kaggle-child-mind-institute-detect-sleep-states"
)
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
