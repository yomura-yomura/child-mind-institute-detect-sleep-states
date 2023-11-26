import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset

this_dir_path = pathlib.Path(__file__).parent
project_root_path = this_dir_path.parent

if __name__ == "__main__":
    exp_name = "ranchantan/exp050-transformer-decoder_retry_resume"

    # inference_step_offsets = np.arange(0, 24, 2) * 12 * 60
    # inference_step_offsets = np.arange(0, 24, 6) * 12 * 60
    # inference_step_offsets = np.arange(0, 24, 4) * 12 * 60
    inference_step_offsets = np.arange(0, 20, 4) * 12 * 60
    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    scores = []
    for i_fold in range(5):
        print(f"fold {i_fold + 1}")
        target_pred_dir_paths = [
            project_root_path
            / "run"
            / "predicted"
            / exp_name
            / ("train" if inference_step_offset <= 0 else f"train-cfg.{inference_step_offset=}")
            / f"fold_{i_fold}"
            for inference_step_offset in inference_step_offsets
        ]
        series_ids = list(
            set(
                p.stem
                for target_pred_dir_path in target_pred_dir_paths
                for p in target_pred_dir_path.glob("*.npz")
            )
        )

        def apply(preds, inference_step_offset):
            df = pd.DataFrame(preds).reset_index(names="step")
            df["step"] += inference_step_offset
            return df.set_index("step")

        preds_df_dict = {
            series_id: pd.concat(
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
            for series_id in tqdm.tqdm(series_ids)
        }

        def get_sub_df(series_id, df):
            preds = pd.concat([df.iloc[:, i::3].mean(axis=1) for i in range(3)], axis=1).to_numpy(
                "f2"
            )
            return cmi_dss_lib.utils.post_process.post_process_for_seg(
                keys=[series_id] * len(preds),
                preds=preds,
                labels=["sleep", "event_onset", "event_wakeup"],
                downsample_rate=2,
                score_th=0.0005,
                distance=96,
            )

        sub_df = pd.concat(
            [get_sub_df(series_id, df) for series_id, df in tqdm.tqdm(preds_df_dict.items())]
        )

        target_event_df = event_df[event_df["series_id"].isin(list(preds_df_dict.keys()))]
        # score = child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(
        #     target_event_df, sub_df
        # )

        score = cmi_dss_lib.utils.metrics.event_detection_ap(target_event_df, sub_df)
        scores.append(score)
        print(f"{score = :.4f}")
        print()
