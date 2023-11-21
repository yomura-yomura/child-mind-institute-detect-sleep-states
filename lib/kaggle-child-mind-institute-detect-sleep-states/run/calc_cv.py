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
exp_name = "ranchantan/exp050-transformer-decoder_retry"

# predicted_fold_dir_path = pathlib.Path("tmp/predicted/ranchantan/exp041/train/fold_0/")
# predicted_dir_path = pathlib.Path("predicted/ranchantan/exp047/train/")
predicted_dir_path = pathlib.Path(f"predicted/{exp_name}/train/")


post_process_modes = {
    # "sleeping_edges_as_probs": cmi_dss_lib.utils.post_process.SleepingEdgesAsProbsSetting(
    #     sleep_prob_th=0.2, min_sleeping_hours=6
    # ),
    # "cutting_probs_by_sleep_prob": cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
    #     watch_interval_hour=6, sleep_occupancy_th=0.3
    # ),
}

# score_th = 0.005
# distance = 96
# score_th = 1e-4
# distance = 88


def calc_score(
    predicted_fold_dir_path,
    score_th: float = 0.005,
    distance: int = 88,
    n_records_per_series_id: int | None = None,
):
    series_ids = [p.stem for p in predicted_fold_dir_path.glob("*.npz")]
    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train").dropna()

    sub_df_list = []
    for series_id in tqdm.tqdm(series_ids):
        preds = np.load(predicted_fold_dir_path / f"{series_id}.npz")["arr_0"]
        sub_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
            [series_id] * len(preds),
            preds,
            downsample_rate=2,
            score_th=score_th,
            distance=distance,
            post_process_modes=post_process_modes,
        )

        if n_records_per_series_id is not None:
            sub_df = (
                sub_df.drop(columns=["row_id"]).sort_values(["score"], ascending=False).head(n_records_per_series_id)
            )
        sub_df_list.append(sub_df)
    sub_df = pd.concat(sub_df_list)
    # sub_df["night"] = sub_df["step"] // (12 * 60 * 24)
    # sub_df = sub_df.sort_values(["score"], ascending=False)
    # sub_df = sub_df.head(400 * len(series_ids))
    # # sub_df = sub_df.groupby(["series_id", "night"]).head(20)
    sub_df = sub_df.sort_values(["series_id", "step"])
    print(sub_df.shape, len(sub_df) / len(series_ids))
    score = cmi_dss_lib.utils.metrics.event_detection_ap(event_df[event_df["series_id"].isin(series_ids)], sub_df)
    # score = child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(
    #     event_df[event_df["series_id"].isin(series_ids)], sub_df
    # )
    return score


if __name__ == "__main__":
    scores = []
    for i_fold in range(5):
        predicted_fold_dir_path = predicted_dir_path / f"fold_{i_fold}"
        score = calc_score(
            predicted_fold_dir_path,
            score_th=1e-4,
            distance=88,
            n_records_per_series_id=1000,
        )
        print(f"fold {i_fold}: {score = :.3f}")
        scores.append(score)
    mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
    print(f"{mean_score_str} ({', '.join(score_strs)})")
