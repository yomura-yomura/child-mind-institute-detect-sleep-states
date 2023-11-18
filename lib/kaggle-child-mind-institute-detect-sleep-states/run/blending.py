import argparse
import os
import pathlib

import cmi_dss_lib.blending
import cmi_dss_lib.utils.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset

np.seterr(all="raise")

project_root_path = pathlib.Path(__file__).parent.parent

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = ["-s", "grid_search"]
else:
    args = None


predicted_npz_format = "predicted-fold_{i_fold}.npz"

post_process_modes = {
    # "sleeping_edges_as_probs": cmi_dss_lib.utils.post_process.SleepingEdgesAsProbsSetting(
    #     sleep_prob_th=0.2, min_sleeping_hours=6
    # ),
    # "cutting_probs_by_sleep_prob": cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
    #     watch_interval_hour=6, sleep_occupancy_th=0.3
    # ),
}


pred_dir_path = project_root_path / "run" / "predicted" / "train"
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"
jumtras_pred_dir_path = project_root_path / "run" / "predicted" / "jumtras"

all_model_dir_path_dict = {
    3: jumtras_pred_dir_path / "exp016-gru-feature-fp16-layer4-ep70-lr-half",
    7: ranchantan_pred_dir_path / "exp015-lstm-feature-108-sigma",
    19: ranchantan_pred_dir_path / "exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma",
    27: jumtras_pred_dir_path / "exp027-TimesNetFeatureExtractor-1DUnet-Unet",
    # 36: ranchantan_pred_dir_path
    # / "exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation",
    41: ranchantan_pred_dir_path / "exp041_retry",
    # 43: jumtras_pred_dir_path / "exp043",
    # 44: ranchantan_pred_dir_path / "exp044-transformer-decoder",
    # 45: ranchantan_pred_dir_path / "exp045-lstm-feature-extractor",
    47: ranchantan_pred_dir_path / "exp047_retry",
    50: ranchantan_pred_dir_path / "exp050-transformer-decoder_retry",
}

weight_dict = {3: 1, 7: 0, 19: 0, 27: 0, 41: 0, 50: 0}  # 17
# weight_dict = {3: 1, 7: 0, 19: 0, 27: 0, 41: 0, 47: 0, 50: 0}  # 18

# score_th = 0.005
# distance = 96
score_th = 0
distance = 88


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-type", "-s", choices=["grid_search", "optuna"], required=True)
    parser.add_argument("--n-cpus", "-n", default=None, type=int)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    args = parser.parse_args(args)

    assert sum(weight_dict.values()) == 1
    print(f"{len(weight_dict) = }")

    model_dir_paths = [all_model_dir_path_dict[i_exp] for i_exp in weight_dict]
    keys_dict, preds_dict = cmi_dss_lib.blending.get_keys_and_preds(model_dir_paths)

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    folds = sorted(map(int, set(args.folds.split(","))))
    print(f"{folds = }")

    def calc_all_scores(
        weights: list[float],
        post_process_modes: dict = None,
        score_th: float = 0.005,
        distance: float = 96,
    ) -> tuple[list[float], list[float]]:
        scores = []
        for i_fold in tqdm.tqdm(folds):
            scores.append(
                cmi_dss_lib.blending.calc_score(
                    i_fold,
                    weights,
                    keys_dict,
                    all_event_df,
                    preds_dict,
                    post_process_modes=post_process_modes,
                    score_th=score_th,
                    distance=distance,
                )
            )

        mean_score_str, *score_strs = map("{:.4f}".format, [np.mean(scores), *scores])
        print(f"{mean_score_str} ({', '.join(score_strs)}) at {weights}")
        return scores, weights

    # calc_all_scores([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2], score_th=score_th, distance=distance)
    # fdas

    models_dir_name = "_".join(str(exp) for exp in weight_dict)

    # weight = get_grid(step=0.1)
    # weight = get_grid(step=0.1, target_sum=1)
    weight = cmi_dss_lib.blending.get_grid(len(model_dir_paths), step=0.1, target_sum=1)
    # weight = get_grid(step=0.02, target_sum=1)

    if folds == [0, 1, 2, 3, 4]:
        models_dir_name = f"blending/{models_dir_name}"
    else:
        models_dir_name = f"blending/{models_dir_name}/{'_'.join(map(str, folds))}"

    record_at_max = cmi_dss_lib.blending.optimize(
        args.search_type, models_dir_name, calc_all_scores, weight, weight_dict, args.n_cpus
    )
    calc_all_scores(record_at_max["weights"], post_process_modes)

# scores = [calc_all_scores(weights=w) for w in tqdm.tqdm(weight, desc="grid search")]


# mean_scores = np.mean(scores, axis=1)
# order = np.argsort(mean_scores)[::-1]
# mean_scores[order]
# weight_list[order]

# #
#
# scores = np.array(
#     [
#         [
#             calc_score(i_fold, weights=[w, 1 - w])
#             for w in tqdm.tqdm(weight_list, desc="grid search")
#         ]
#         for i_fold in range(5)
#     ]
# )  # (fold, weight)
#
# weights = weight_list[np.argmax(scores, axis=1)]
# np.mean(np.max(scores, axis=1))
