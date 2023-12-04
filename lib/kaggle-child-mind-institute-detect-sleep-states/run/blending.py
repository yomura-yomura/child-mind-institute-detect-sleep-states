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

# np.seterr(all="raise")

project_root_path = pathlib.Path(__file__).parent.parent

if os.environ.get("RUNNING_INSIDE_PYCHARM", False):
    args = [
        # "-s", "grid_search", "-f", "0"
    ]
else:
    args = None


predicted_npz_format = "predicted-fold_{i_fold}.npz"

post_process_modes = {
    # "sleeping_edges_as_probs": cmi_dss_lib.utils.post_process.SleepingEdgesAsProbsSetting(
    #     sleep_prob_th=0.2, min_sleeping_hours=6
    # ),
    # "cutting_probs_by_sleep_prob": cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
    #     watch_interval_hour=7.5, sleep_occupancy_th=0.03, version=0, n_continuous=0
    # ),
    # "cutting_probs_by_sleep_prob": dict(
    #     onset=cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
    #         watch_interval_hour=7.5,
    #         sleep_occupancy_th=0.04,
    #     ),
    #     wakeup=cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
    #         watch_interval_hour=6.0,
    #         sleep_occupancy_th=0.03,
    #     ),
    # )
    # "cutting_probs_on_repeating": cmi_dss_lib.utils.post_process.CuttingProbsOnRepeating(
    #     prepare_data_dir_path=f"../output/prepare_data/train/robust_scaler",
    #     interval_th=15 * 60 // 5,
    # ),
}


pred_dir_path = project_root_path / "run" / "predicted" / "train"
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"
jumtras_pred_dir_path = project_root_path / "run" / "predicted" / "jumtras"
stacking_pred_dir_path = project_root_path / "run" / "predicted" / "train_stacking"
blending_pred_dir_path = project_root_path / "run" / "predicted" / "blending"


exp_name_dict = {
    "3": "jumtras/exp016-gru-feature-fp16-layer4-ep70-lr-half",
    "7": "ranchantan/exp015-lstm-feature-108-sigma",
    "19": "ranchantan/exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma",
    "27": "jumtras/exp027-TimesNetFeatureExtractor-1DUnet-Unet",
    "41": "ranchantan/exp041_retry",
    "47": "ranchantan/exp047_retry",
    "50": "ranchantan/exp050-transformer-decoder_retry_resume",
    "52": "jumtras/exp052",
    "53": "jumtras/exp053",
    "54": "ranchantan/exp054",
    "55": "ranchantan/exp055",
    "58": "jumtras/exp058",
    "60": "ranchantan/exp060",
    "73": "ranchantan/exp073_resume",
    "75": "ranchantan/exp075-wakeup_5",
}

all_model_dir_path_dict = (
    {
        "3": jumtras_pred_dir_path / "exp016-gru-feature-fp16-layer4-ep70-lr-half",
        "7": ranchantan_pred_dir_path / "exp015-lstm-feature-108-sigma",
        "19": ranchantan_pred_dir_path / "exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma",
        "27": jumtras_pred_dir_path / "exp027-TimesNetFeatureExtractor-1DUnet-Unet",
        "41": ranchantan_pred_dir_path / "exp041_retry",
        "47": ranchantan_pred_dir_path / "exp047_retry",
        "50": ranchantan_pred_dir_path / "exp050-transformer-decoder_retry_resume",
        "52": jumtras_pred_dir_path / "exp052",
        "53": jumtras_pred_dir_path / "exp053",
        "54": ranchantan_pred_dir_path / "exp054",
        "55": ranchantan_pred_dir_path / "exp055",
        "58": jumtras_pred_dir_path / "exp058",
        "60": ranchantan_pred_dir_path / "exp060",
        "73": ranchantan_pred_dir_path / "exp073_resume",
        "75": ranchantan_pred_dir_path / "exp075-wakeup_5",
        "85": jumtras_pred_dir_path / "exp085",
        "88": jumtras_pred_dir_path / "exp088",
        "99": ranchantan_pred_dir_path / "exp099_resume",
        "100": ranchantan_pred_dir_path / "exp100",
        "101": ranchantan_pred_dir_path / "exp101",
        "105": ranchantan_pred_dir_path / "exp105",
        "107": ranchantan_pred_dir_path / "exp107",
        "109": pred_dir_path / "exp109",
        "110": ranchantan_pred_dir_path / "exp110",
    }
    | {
        "s6": stacking_pred_dir_path / "s_exp006",
    }
    | {
        "b26": blending_pred_dir_path / "exp026",
        "b28": blending_pred_dir_path / "exp028",
        # "b29": blending_pred_dir_path / "exp029",
        "b30": blending_pred_dir_path / "exp030",
        "b32": blending_pred_dir_path / "exp032",
        "b33": blending_pred_dir_path / "exp033",
        "b34": blending_pred_dir_path / "exp034",
    }
)

# weight_dict = {3: 1, 7: 0, 19: 0, 27: 0, 41: 0, 50: 0}  # 17
# weight_dict = {3: 1, 7: 0, 19: 0, 27: 0, 41: 0, 47: 0, 50: 0}  # 18
# weight_dict = {
#     "27": 0.1,
#     "41": 0.1,
#     "47": 0.1,
#     "50": 0.2,
#     "52": 0.2,
#     "53": 0.3,
# }  # 19
# weight_dict = {7: 1, 19: 0, 27: 0, 47: 0, 50: 0, 52: 0, 53: 0}  # 20
# weight_dict = {"19": 0.1, "27": 0.1, "47": 0.1, "50": 0.2, "52": 0.2, "53": 0.3}  # 20
# weight_dict = {47: 1, 50: 0, 52: 0, 53: 0}  # 21
# weight_dict = {19: 0.1, 27: 0.1, 50: 0.2, 53: 0.3, 55: 0.1, 58: 0.2}  # 22
# weight_dict = {3: 1, 19: 0, 50: 0, 53: 0, 54: 0, 55: 0}
# weight_dict = {3: 1, 50: 0, 52: 0, 53: 0, 55: 0, 58: 0}
# weight_dict = {"19": 0.1, "27": 0.1, "50": 0.2, "53": 0.3, "55": 0.1, "58": 0.2}  # 23
# weight_dict = {"19": 0.1, "50": 0.2, "53": 0.2, "s6": 0.3, "58": 0.2}

# weight_dict = {"19": 0.1, "50": 0.2, "53": 0.3, "58": 0.1, "85": 0.2, "88": 0.1}
# weight_dict = {"19": 0.05, "27": 0.05, "50": 0.2, "53": 0.3, "58": 0.1, "85": 0.2, "88": 0.1}
# weight_dict = {
#     # "7": 0.05,
#     "19": 0.05,
#     "27": 0.05,
#     "50": 0.2,
#     "53": 0.3,
#     "58": 0.1,
#     "85": 0.2,
#     "88": 0.1,
# }  # 25
# weight_dict = {"19": 0.1, "50": 0.2, "53": 0.3, "58": 0.1, "85": 0.2, "88": 0.1}  # 24

# weight_dict = {"50": 0.1, "53": 0.2, "58": 0.1, "85": 0.2, "88": 0.2, "100": 0.2}  # 26
# weight_dict = {"50": 0.2, "53": 0.2, "85": 0.2, "88": 0.2, "100": 0.2}  # 27
# weight_dict = {
#     "19": 0.05,
#     "50": 0.1,
#     "53": 0.2,
#     "58": 0.05,
#     "85": 0.2,
#     "88": 0.2,
#     "100": 0.2,
# }  # 28

# weight_dict = {
#     "50": 0.085,
#     "53": 0.17,
#     "58": 0.085,
#     "85": 0.17,
#     "88": 0.17,
#     "100": 0.17,
#     "101": 0.15,
# }  # 29

# weight_dict = {
#     "19": 0.045,
#     "50": 0.09,
#     "53": 0.18,
#     "58": 0.045,
#     "85": 0.18,
#     "88": 0.18,
#     "100": 0.18,
#     "101": 0.1,
# }  # 30

# weight_dict = {
#     "19": 0.0405,
#     "50": 0.081,
#     "53": 0.162,
#     "58": 0.0405,
#     "85": 0.162,
#     "88": 0.162,
#     "99": 0.1,
#     "100": 0.162,
#     "101": 0.09,
# }  # 31

# weight_dict = {
#     "19": 0.03864959320010797,
#     "50": 0.08553410217905172,
#     "53": 0.16561358858285558,
#     "58": 0.030066808112389563,
#     "85": 0.14162568745283677,
#     "88": 0.17845648941370792,
#     "99": 0.11488021220066767,
#     "100": 0.16421709597493192,
#     "101": 0.08095642288345084,
# }  # 32


# weight_dict = {
#     "19": 0.03710360947210365,
#     "50": 0.08211273809188965,
#     "53": 0.15898904503954134,
#     "58": 0.02886413578789398,
#     "85": 0.13596065995472328,
#     "88": 0.1713182298371596,
#     "99": 0.15028500371264097,
#     "100": 0.15764841213593464,
#     "101": 0.0777181659681128,
# }  # 33


# limit_dict = {
#     "19": (0, 0.1),
#     "50": (0, 0.1),
#     "53": (0.1, 0.2),
#     "58": (0, 0.1),
#     "85": (0.1, 0.2),
#     "88": (0.1, 0.2),
#     "99": (0, 0.2),
#     "100": (0.1, 0.2),
#     "101": (0, 0.15),
#     "107": (0.5, 0.15),
# }

# weight_dict = {"b26": 0.85, "101": 0.15}
# weight_dict = {"b28": 0.9, "101": 0.1}
# weight_dict = {"b30": 0.9, "99": 0.1}
# weight_dict = {"b32": 0.96, "99": 0.04}
# weight_dict = {"b33": 0.88, "107": 0.12}
# weight_dict = {
#     "19": 0.03265117633545121,
#     "50": 0.0722592095208629,
#     "53": 0.13991035963479637,
#     "58": 0.0254004394933467,
#     "85": 0.11964538076015649,
#     "88": 0.15076004225670045,
#     "99": 0.13225080326712405,
#     "100": 0.13873060267962248,
#     "101": 0.06839198605193927,
#     "107": 0.12,
# }  # 34

weight_dict = {"b34": 1, "110": 0}

limit_dict = {
    "19": (0, 0.04),
    "50": (0, 0.1),
    "53": (0.1, 0.2),
    "58": (0, 0.05),
    "85": (0.05, 1.5),
    "88": (0.1, 0.2),
    "99": (0.05, 0.15),
    "100": (0.05, 0.15),
    "101": (0, 0.1),
    "107": (0.05, 0.15),
}

# weight_dict = {"19": 0.1, "50": 0.2, "53": 0.3, "58": 0.1, "85": 0.2, "88": 0.1, "100": 0}

# score_th = 0.005
# distance = 96
score_th = 1e-4
distance = 88


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-type", "-s", choices=["grid_search", "optuna"], default=None)
    parser.add_argument("--n-cpus", "-n", default=None, type=int)
    parser.add_argument("--folds", "-f", type=str, default="0,1,2,3,4")
    args = parser.parse_args(args)

    weight_dict = {str(k): v for k, v in weight_dict.items()}
    assert np.isclose(sum(weight_dict.values()), 1), sum(weight_dict.values())
    print(f"{len(weight_dict) = }")

    folds = sorted(map(int, set(args.folds.split(","))))
    print(f"{folds = }")

    model_dir_paths = [all_model_dir_path_dict[exp] for exp in weight_dict]
    if limit_dict is None:
        limits = [(0, 1) for _ in weight_dict]
    else:
        limits = [limit_dict.get(exp, (0, 1)) for exp in weight_dict]

    keys_dict, preds_dict = cmi_dss_lib.blending.get_keys_and_preds(model_dir_paths, folds)

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    def calc_all_scores(
        weights: list[float],
        post_process_modes: dict = None,
        score_th: float = score_th,
        distance: float = distance,
        calc_type: str = "fast",
        n_records_per_series_id: int = None,
        print_msg: bool = False,
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
                    calc_type=calc_type,
                    n_records_per_series_id=n_records_per_series_id,
                    print_msg=print_msg,
                )
            )

        mean_score_str, *score_strs = map("{:.4f}".format, [np.mean(scores), *scores])
        print(f"{mean_score_str} ({', '.join(score_strs)}) at {weights}")
        return scores, weights

    if args.search_type is None:
        print(f"calc score for {weight_dict}")
        calc_all_scores(
            list(weight_dict.values()),
            # score_th=0.005,
            # distance=96,
            score_th=1e-4,
            distance=88,
            # n_records_per_series_id=1000,
            n_records_per_series_id=2000,
            # n_records_per_series_id=None,
            post_process_modes=post_process_modes,
            print_msg=True,
            calc_type="normal",
        )
    else:
        models_dir_name = "_".join(str(exp) for exp in weight_dict)

        # weight = cmi_dss_lib.blending.get_grid(len(model_dir_paths), step=0.1, target_sum=1)
        # weight = cmi_dss_lib.blending.get_grid(len(model_dir_paths), step=0.05, target_sum=1)
        if args.search_type == "grid_search":
            weight = cmi_dss_lib.blending.get_grid(
                len(model_dir_paths), step=0.04, target_sum=1, limits=limits
            )
        else:
            weight = None

        if folds == [0, 1, 2, 3, 4]:
            models_dir_name = f"blending/{models_dir_name}"
        else:
            models_dir_name = f"blending/{models_dir_name}/{'_'.join(map(str, folds))}"

        import optuna

        def objective(trial: optuna.trial.FrozenTrial) -> float:
            weights = []
            for i in range(len(weight_dict) - 1):
                weight = trial.suggest_float(f"w{i}", *limits[i])
                weights.append(weight)
            weight = 1 - sum(weights)
            weights.append(weight)

            trial.set_user_attr("constraint", (1 - sum(weights), sum(weights) - 1))

            min_w, max_w = limits[-1]
            if not ((min_w <= weight) and (weight <= max_w)):
                return np.nan

            print(f"{weights = }")
            scores, _ = calc_all_scores(weights)
            score = float(np.mean(scores))
            print(f"{score = }")
            return score

        record_at_max = cmi_dss_lib.blending.optimize(
            args.search_type,
            models_dir_name,
            calc_all_scores,
            weight,
            limits,
            weight_dict,
            args.n_cpus,
            objective,
        )
        calc_all_scores(
            record_at_max["weights"],
            post_process_modes,
            print_msg=True,
            score_th=1e-4,
            distance=88,
            n_records_per_series_id=2000,
            calc_type="normal",
        )
