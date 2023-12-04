import pathlib
import shutil
from typing import Literal

import numpy as np
import pandas as pd
import tqdm


def replace_predicted_pred_dirs_with_mean_preds_dir(
    predicted_dir_path: pathlib.Path,
    model_dir_path: pathlib.Path,
    phase: Literal["train", "test", "dev"],
):
    predicted_phase_dir_path = predicted_dir_path / pathlib.Path(*model_dir_path.parts[-2:]) / phase

    predicted_fold_dir_paths = sorted(predicted_phase_dir_path.glob("fold_*"))
    assert len(predicted_fold_dir_paths) == 5

    common_unique_series_ids = None
    for predicted_fold_dir_path in predicted_fold_dir_paths:
        unique_series_ids = sorted(p.stem for p in predicted_fold_dir_path.glob("*.npz"))
        if common_unique_series_ids is None:
            common_unique_series_ids = unique_series_ids
        else:
            assert common_unique_series_ids == unique_series_ids
    assert len(common_unique_series_ids) > 0

    predicted_mean_dir_path = predicted_phase_dir_path / "mean"
    predicted_mean_dir_path.mkdir(exist_ok=True)

    for series_id in tqdm.tqdm(common_unique_series_ids, desc="averaging preds for each series_id"):
        mean_preds = None
        for predicted_fold_dir_path in predicted_fold_dir_paths:
            preds = np.load(predicted_fold_dir_path / f"{series_id}.npz")["arr_0"]
            if mean_preds is None:
                mean_preds = preds
            else:
                mean_preds += preds
        mean_preds /= 5
        np.savez_compressed(predicted_mean_dir_path / f"{series_id}.npz", mean_preds)

    for predicted_fold_dir_path in predicted_fold_dir_paths:
        shutil.rmtree(predicted_fold_dir_path)


def save_overlapping_mean_as_npz(
    predicted_dir_path: pathlib.Path,
    model_dir_path: pathlib.Path,
    i_fold: int,
    phase: str,
    inference_step_offsets: list[int],
):
    exp_name = "/".join(model_dir_path.parts[-2:])
    target_pred_dir_paths = [
        predicted_dir_path / exp_name / f"{phase}-cfg.{inference_step_offset=}" / f"fold_{i_fold}"
        for inference_step_offset in inference_step_offsets
    ]
    series_ids = list(
        set(p.stem for target_pred_dir_path in target_pred_dir_paths for p in target_pred_dir_path.glob("*.npz"))
    )

    def apply(preds, inference_step_offset):
        df = pd.DataFrame(preds).reset_index(names="step")
        df["step"] += inference_step_offset
        return df.set_index("step")

    target_pred_mean_dir_path = predicted_dir_path / exp_name / f"{phase}" / f"fold_{i_fold}"
    target_pred_mean_dir_path.mkdir(exist_ok=True, parents=True)
    for series_id in tqdm.tqdm(series_ids, desc="save overlapping-mean npz"):
        df = pd.concat(
            [
                apply(
                    np.load(target_pred_dir_path / f"{series_id}.npz")["arr_0"],
                    inference_step_offset,
                )
                for target_pred_dir_path, inference_step_offset in zip(
                    target_pred_dir_paths, inference_step_offsets, strict=True
                )
            ],
            axis=1,
        )
        preds = pd.concat([df.iloc[:, i::3].mean(axis=1) for i in range(3)], axis=1).to_numpy("f2")
        np.savez_compressed(target_pred_mean_dir_path / f"{series_id}.npz", preds)

    for p in target_pred_dir_paths:
        shutil.rmtree(p)
