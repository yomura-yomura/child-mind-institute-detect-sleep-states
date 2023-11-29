import os
import pathlib
from typing import Literal

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm

from ..data.utils import get_duration_dict


def validation(
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
):
    predicted_model_dir_paths = list(predicted_dir_path.glob(f"*/*/{phase}"))
    assert len(predicted_model_dir_paths) > 0

    series_id_grouped_npz_dir_paths = []
    for i_fold in range(5):
        model_paths = []
        for predicted_model_dir_path in predicted_model_dir_paths:
            model_paths.append(predicted_model_dir_path / f"fold_{i_fold}")
        assert len(model_paths) > 0
        series_id_grouped_npz_dir_paths.append(model_paths)

    common_unique_series_ids_list = []
    for i_fold in tqdm.trange(5):
        common_unique_series_ids = None
        for i_model in range(len(predicted_model_dir_paths)):
            unique_series_ids = sorted(p.stem for p in series_id_grouped_npz_dir_paths[i_fold][i_model].glob("*.npz"))
            assert len(unique_series_ids) == len(set(unique_series_ids))

            if common_unique_series_ids is None:
                common_unique_series_ids = unique_series_ids
            else:
                assert common_unique_series_ids == unique_series_ids
        assert common_unique_series_ids is not None
        common_unique_series_ids_list.append(common_unique_series_ids)

    min_duration_dict = get_duration_dict(phase)

    scores = []
    for i_fold in range(5):
        print(f"fold {i_fold}")
        sub_df_list = []
        for series_id in tqdm.tqdm(common_unique_series_ids_list[i_fold], desc="summing up preds for each series_id"):
            preds_list = []
            weights = []
            for i_model, pred_dir_path in enumerate(predicted_model_dir_paths):
                preds_list.append(
                    np.load(series_id_grouped_npz_dir_paths[i_fold][i_model] / f"{series_id}.npz")["arr_0"][
                        : min_duration_dict[series_id]
                    ]
                )
                weights.append(
                    model_dir_path_info_dict[
                        os.path.join("/kaggle/input/cmi-dss-ensemble-models/", *pred_dir_path.parts[-3:-1])
                    ][0]
                )

            preds = np.average(preds_list, weights=weights, axis=0)

            sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
                series_id=series_id,
                preds=preds,
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

        event_df = pd.read_csv("/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv").dropna()

        score = cmi_dss_lib.utils.metrics.event_detection_ap(
            event_df[event_df["series_id"].isin(common_unique_series_ids_list[i_fold])], sub_df
        )
        print(f"{score = }")
        scores.append(score)

    mean_score_str, *score_strs = map("{:.4f}".format, (np.mean(scores), *scores))
    print(f"{mean_score_str} ({', '.join(score_strs)})")
