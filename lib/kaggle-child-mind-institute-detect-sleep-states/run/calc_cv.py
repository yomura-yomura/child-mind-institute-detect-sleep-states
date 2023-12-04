import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.score

# np.seterr(all="raise")

# exp_name = "ranchantan/exp041"
# exp_name = "ranchantan/exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma"
# exp_name = "ranchantan/exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation"
# exp_name = "jumtras/exp027-TimesNetFeatureExtractor-1DUnet-Unet"
# exp_name = "jumtras/exp043"
# exp_name = "ranchantan/exp050-transformer-decoder_retry"
# exp_name = "ranchantan/exp050-transformer-decoder_retry_resume"
# exp_name = "combined/exp050_exp75-wakeup"
# exp_name = "ranchantan/exp075-wakeup_6"
# exp_name = "ranchantan/exp099_resume"

# exp_name = "blending/exp026"
# exp_name = "blending/exp028"
# exp_name = "blending/exp029"
exp_name = "blending/exp032"

# exp_name = "train/exp101"

# predicted_fold_dir_path = pathlib.Path("tmp/predicted/ranchantan/exp041/train/fold_0/")
# predicted_dir_path = pathlib.Path("predicted/ranchantan/exp047/train/")
predicted_dir_path = pathlib.Path(f"predicted/{exp_name}/train/")

# score_th = 0.005
# distance = 96
# score_th = 1e-4
# distance = 88


def calc_score(
    predicted_fold_dir_path,
    labels,
    downsample_rate,
    score_th: float = 0.005,
    distance: int = 96,
    calc_type: str = "fast",
    n_records_per_series_id: int | None = None,
    post_process_modes=None,
):
    series_ids = [p.stem for p in predicted_fold_dir_path.glob("*.npz")]
    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train").dropna()
    event_df = event_df[event_df["series_id"].isin(series_ids)]
    event_df = event_df[event_df["event"].isin([event for event in ["onset", "wakeup"] if f"event_{event}" in labels])]

    import cmi_dss_lib.data.utils

    start_timing_dict = cmi_dss_lib.data.utils.get_start_timing_dict("train")

    sub_df_list = []
    for series_id in tqdm.tqdm(series_ids):
        preds = np.load(predicted_fold_dir_path / f"{series_id}.npz")["arr_0"]

        sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
            series_id,
            preds,
            labels=list(labels),
            downsample_rate=downsample_rate,
            score_th=score_th,
            distance=distance,
            post_process_modes=post_process_modes,
            start_timing_dict=start_timing_dict,
        )
        if n_records_per_series_id is not None:
            sub_df = sub_df.sort_values(["score"], ascending=False).head(n_records_per_series_id)
        print(f"{len(sub_df) = }")
        sub_df_list.append(sub_df)
    sub_df = pd.concat(sub_df_list)
    # sub_df["night"] = sub_df["step"] // (12 * 60 * 24)
    # sub_df = sub_df.sort_values(["score"], ascending=False)
    # sub_df = sub_df.head(400 * len(series_ids))
    # # sub_df = sub_df.groupby(["series_id", "night"]).head(20)
    sub_df = sub_df.sort_values(["series_id", "step"])

    if calc_type == "fast":
        score = child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(event_df, sub_df)
    elif calc_type == "normal":
        score = cmi_dss_lib.utils.metrics.event_detection_ap(event_df, sub_df)
    else:
        raise ValueError(f"unexpected {calc_type=}")
    return score


if __name__ == "__main__":
    if not predicted_dir_path.exists():
        raise FileNotFoundError(predicted_dir_path)

    post_process_modes = cmi_dss_lib.utils.post_process.PostProcessModeWithSetting(
        # "sleeping_edges_as_probs": cmi_dss_lib.utils.post_process.SleepingEdgesAsProbsSetting(
        #     sleep_prob_th=0.2, min_sleeping_hours=6
        # ),
        cutting_probs_by_sleep_prob=cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
            watch_interval_hour=7.5, sleep_occupancy_th=0.03, version=1, n_continuous=12 * 60 * 5
        ),
        # cutting_probs_by_sleep_prob=cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSettingByEvent(
        #     onset=cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
        #         watch_interval_hour=7.5,
        #         sleep_occupancy_th=0.04,
        #     ),
        #     wakeup=cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
        #         watch_interval_hour=6.0,
        #         sleep_occupancy_th=0.03,
        #     ),
        # ),
        # cutting_probs_on_repeating=cmi_dss_lib.utils.post_process.CuttingProbsOnRepeating(
        #     prepare_data_dir_path="../output/prepare_data/train/robust_scaler",
        #     interval_th=15 * 60 // 5,
        # ),
        # average_submission_over_steps=cmi_dss_lib.utils.post_process.AveragingSubmissionOverSteps(
        #     interval=12 * 10
        # ),
    )

    scores = []
    for i_fold in range(5):
        predicted_fold_dir_path = predicted_dir_path / f"fold_{i_fold}"
        score = calc_score(
            predicted_fold_dir_path,
            labels=["sleep", "event_onset", "event_wakeup"],
            # score_th=1e-4,
            # distance=88,
            score_th=1e-5,
            distance=95,
            n_records_per_series_id=2000,
            # score_th=0.005,
            # distance=96,
            downsample_rate=2,
            calc_type="normal",
            post_process_modes=post_process_modes,
        )
        print(f"fold {i_fold}: {score = :.4f}")
        scores.append(score)
    mean_score_str, *score_strs = map("{:.4f}".format, [np.mean(scores), *scores])
    print(f"{mean_score_str} ({', '.join(score_strs)})")
