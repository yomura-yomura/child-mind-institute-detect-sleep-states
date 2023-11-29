import pathlib
from typing import Literal

import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm

from ..data.utils import get_duration_dict


def create_submission_csv(
    model_dir_path_info_dict: dict,
    predicted_dir_path: pathlib.Path,
    *,
    phase: Literal["test", "dev"],
    use_stacking: bool,
    downsample_rate: int,
    score_th: float,
    distance: int,
    post_process_modes: cmi_dss_lib.utils.post_process.PostProcessModeWithSetting,
    n_records_per_series_id: int,
) -> None:
    predicted_mean_dir_paths = list(predicted_dir_path.glob(f"*/*/{phase}/mean"))
    assert len(predicted_mean_dir_paths) > 0

    common_unique_series_ids: list[str] | None = None
    for predicted_mean_dir_path in predicted_mean_dir_paths:
        unique_series_ids = sorted(p.stem for p in predicted_mean_dir_path.glob("*.npz"))
        if common_unique_series_ids is None:
            common_unique_series_ids = unique_series_ids
        else:
            assert common_unique_series_ids == unique_series_ids
    assert common_unique_series_ids is not None

    min_duration_dict = get_duration_dict(phase)

    sub_df_list = []
    for series_id in tqdm.tqdm(common_unique_series_ids, desc="create submissions"):
        blended_preds = None
        for i_model, predicted_mean_dir_path in enumerate(predicted_mean_dir_paths):
            folded_preds = np.load(predicted_mean_dir_path / f"{series_id}.npz")["arr_0"][
                : min_duration_dict[series_id]
            ]
            if not use_stacking:
                weight = model_dir_path_info_dict[
                    "/kaggle/input/cmi-dss-ensemble-models/"
                    + "/".join(predicted_mean_dir_path.parts[-4:-2])
                ][0]
                folded_preds *= weight

            if blended_preds is None:
                blended_preds = folded_preds
            else:
                blended_preds += folded_preds

        if not use_stacking:
            blended_preds /= sum(weight for weight, _ in model_dir_path_info_dict.values())

        sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
            series_id=series_id,
            preds=blended_preds,
            labels=["sleep", "event_onset", "event_wakeup"],
            downsample_rate=downsample_rate,
            score_th=score_th,
            distance=distance,
            post_process_modes=post_process_modes,
            n_records_per_series_id=n_records_per_series_id,
            print_msg=False,
        )

        sub_df_list.append(sub_df)

    sub_df = pd.concat(sub_df_list)
    sub_df = sub_df.sort_values(["series_id", "step"])
    sub_df = sub_df.dropna()  # just to be sure
    sub_df = sub_df.reset_index(drop=True).reset_index(names="row_id")
    sub_df.to_csv("/kaggle/working/submission.csv", index=False)
