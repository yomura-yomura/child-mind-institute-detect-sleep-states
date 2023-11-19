import argparse
import multiprocessing
import os
import pathlib

import cmi_dss_lib.utils.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray

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
    "cutting_probs_by_sleep_prob": cmi_dss_lib.utils.post_process.CuttingProbsBySleepProbSetting(
        watch_interval_hour=6, sleep_occupancy_th=0.3
    ),
}


pred_dir_path = project_root_path / "run" / "predicted" / "train"
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"
jumtras_pred_dir_path = project_root_path / "run" / "predicted" / "jumtras"

all_model_dir_path_dict = {
    3: jumtras_pred_dir_path / "exp016-gru-feature-fp16-layer4-ep70-lr-half",  # 3
    7: ranchantan_pred_dir_path / "exp015-lstm-feature-108-sigma",  # 7
    19: ranchantan_pred_dir_path / "exp019-stacked-gru-4-layers-24h-duration-4bs-108sigma",
    27: jumtras_pred_dir_path / "exp027-TimesNetFeatureExtractor-1DUnet-Unet",
    36: ranchantan_pred_dir_path
    / "exp036-stacked-gru-4-layers-24h-duration-4bs-108sigma-with-step-validation",
    41: ranchantan_pred_dir_path / "exp041",
    # 43: jumtras_pred_dir_path / "exp043",
    44: ranchantan_pred_dir_path / "exp044-transformer-decoder",
    45: ranchantan_pred_dir_path / "exp045-lstm-feature-extractor",
    47: ranchantan_pred_dir_path / "exp047",
}

# weight_dict = {7: 0.2, 19: 0.3, 27: 0.3, 41: 0.2}  # 12
# weight_dict = {7: 0.2, 19: 0.3, 27: 0.2, 41: 0.15, 45: 0.15}
# weight_dict = {3: 1, 7: 0, 19: 0, 41: 0, 27: 0}
# weight_dict = {3: 1, 19: 0, 27: 0, 41: 0, 44: 0, 45: 0}
# weight_dict = {3: 1, 19: 0, 27: 0, 41: 0, 44: 0, 47: 0}
# weight_dict = {3: 1, 19: 0, 27: 0, 41: 0, 44: 0, 45: 0, 47: 0}
weight_dict = {3: 1, 7: 0, 19: 0, 36: 0, 44: 0, 45: 0, 47: 0}

assert sum(weight_dict.values()) == 1
print(f"{len(weight_dict) = }")

model_dir_paths = [all_model_dir_path_dict[i_exp] for i_exp in weight_dict]


def calc_score(
    i_fold: int,
    weights: list[int],
    keys_dict,
    all_event_df,
    preds_dict,
    post_process_modes,
    score_th=0.005,
    distance=96,
):
    series_ids = keys_dict[i_fold]
    # unique_series_ids = np.unique([str(k).split("_")[0] for k in keys])
    unique_series_ids = np.unique(series_ids)
    event_df = all_event_df[all_event_df["series_id"].isin(unique_series_ids)]

    df_submit_list = []
    for series_id, preds in zip(series_ids, preds_dict[i_fold], strict=True):
        assert preds.shape[0] == len(weights), (preds.shape, len(weights))
        mean_preds = np.average(preds, axis=0, weights=weights)
        df_submit_list.append(
            cmi_dss_lib.utils.post_process.post_process_for_seg(
                #
                preds=mean_preds,
                # preds=corrected_preds[:, :, [1, 2]],
                downsample_rate=2,
                keys=[series_id] * len(mean_preds),
                score_th=score_th,
                distance=distance,
                post_process_modes=post_process_modes,
                print_msg=False,
            )
        )
    df_submit = pd.concat(df_submit_list)

    return cmi_dss_lib.utils.metrics.event_detection_ap(
        event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
    )


# scores = calc_all_scores(weights=[1, 1])


def get_grid(step: float, target_sum: float = 1, start: float = 0) -> NDArray[np.float_]:
    assert step < 1
    assert 0 <= target_sum

    target_sum *= round(1 / step)
    base_weight = pd.DataFrame(
        np.arange(round(1 / step) + 1) + round(start * round(1 / step)), dtype="i4"
    )

    weight = base_weight.copy()
    for i, _ in enumerate(tqdm.trange(len(model_dir_paths) - 1)):
        weight = weight.merge(base_weight.rename(columns={0: i + 1}), how="cross")
        weight = weight[np.sum(weight, axis=1) <= target_sum].reset_index(drop=True)
    weight = weight.to_numpy()
    weight = weight[np.sum(weight, axis=1) == target_sum]
    print(f"{weight.shape = }")
    return weight * step


def get_keys_and_preds(model_dir_paths: list[pathlib.Path | str]):
    predicted_npz_dir_paths = [
        [
            pathlib.Path(model_dir_path) / "train" / f"fold_{i_fold}"
            for model_dir_path in model_dir_paths
        ]
        for i_fold in range(5)
    ]  # (fold, model)
    for predicted_npz_dir_paths_by_fold in predicted_npz_dir_paths:
        for path in predicted_npz_dir_paths_by_fold:
            if not path.exists():
                raise FileNotFoundError(path)

    count_by_series_id_df = (
        child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df(
            "train", as_polars=True
        )
        .group_by("series_id")
        .count()
        .collect()
    )
    min_duration_dict = dict(count_by_series_id_df.iter_rows())

    keys_dict = {}
    preds_dict = {}
    for i_fold in tqdm.trange(5):
        (
            keys_dict[i_fold],
            preds_dict[i_fold],
        ) = cmi_dss_lib.utils.common.load_predicted_npz_group_by_series_id(
            predicted_npz_dir_paths[i_fold], min_duration_dict
        )
    return keys_dict, preds_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-type", "-s", choices=["grid_search", "optuna"], required=True)
    parser.add_argument("--n-cpus", "-n", default=8, type=int)
    args = parser.parse_args(args)

    keys_dict, preds_dict = get_keys_and_preds(model_dir_paths)
    # keys_dict, preds_dict = get_keys_and_preds(list(model_dir_path_dict.values()))

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    # calc_score(0, [1, 1, 1], keys_dict, all_event_df, preds_dict, None)

    def calc_all_scores(weights: list[int], post_process_modes=None):
        scores = []
        for i_fold in tqdm.trange(5):
            scores.append(
                calc_score(
                    i_fold,
                    weights,
                    keys_dict,
                    all_event_df,
                    preds_dict,
                    post_process_modes=post_process_modes,
                )
            )

        mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
        print(f"{mean_score_str} ({', '.join(score_strs)})")
        return scores, weights

    models_dir_name = "_".join(str(exp) for exp in weight_dict)

    match args.search_type:
        case "grid_search":
            # weight = get_grid(step=0.1)
            # weight = get_grid(step=0.1, target_sum=1)
            weight = get_grid(step=0.1, target_sum=1)
            # weight = get_grid(step=0.02, target_sum=1)

            target_csv_path = (
                pathlib.Path(__file__).parent / "grid_search" / models_dir_name / "grid_search.csv"
            )

            if target_csv_path.exists():
                df = pd.read_csv(target_csv_path)
                df["scores"] = df["scores"].apply(
                    lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")]
                )
                df["weights"] = df["weights"].apply(
                    lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")]
                )

                # if True:
                #     target_weight = df.iloc[df["CV"].argmax()]["weights"]
                #     print(f"{target_weight = }")
                #     new_weight = target_weight + get_grid(0.05, 0, -0.5)
                #     new_weight = new_weight[np.all(new_weight > 0, axis=1)]
                #     assert np.all(np.isclose(np.sum(new_weight, axis=1), 1))
                #     weight = np.concatenate([weight, new_weight], axis=0)

                loaded_weight = np.array(df["weights"].tolist())
                weight = np.array(
                    [w for w in weight if not np.any(np.all(np.isclose(w, loaded_weight), axis=1))]
                )
                print(f"-> {weight.shape = }")

                records = df.to_dict("records")
            else:
                records = []

            n_steps_to_save = 30
            with multiprocessing.Pool(args.n_cpus) as p:
                with tqdm.tqdm(total=len(weight), desc="grid search") as t:
                    for scores, weights in p.imap_unordered(calc_all_scores, weight.tolist()):
                        t.update(1)
                        records.append(
                            {"CV": np.mean(scores), "scores": scores, "weights": weights}
                        )

                        if len(records) % n_steps_to_save == 0:
                            df = pd.DataFrame(records)
                            target_csv_path.parent.mkdir(parents=True, exist_ok=True)
                            df.to_csv(target_csv_path, index=False)

                            record_at_max = df.iloc[df["CV"].argmax()]
                            print(
                                f"""
max:
CV = {record_at_max["CV"]:.4f}
weights = {dict(zip(weight_dict, record_at_max["weights"]))}
"""
                            )

            df = pd.DataFrame(records)
            target_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target_csv_path, index=False)

            record_at_max = df.iloc[df["CV"].argmax()]
            print(
                f"""
max:
CV = {record_at_max["CV"]:.4f}
weights = {dict(zip(weight_dict, record_at_max["weights"]))}
"""
            )
            print(dict(zip(model_dir_paths, record_at_max["weights"])))

            score = calc_all_scores(record_at_max["weights"], post_process_modes)
        case "optuna":
            import optuna

            def objective(trial: optuna.Trial):
                total = 0

                weights = []
                for i in range(len(model_dir_paths) - 1):
                    weight = trial.suggest_float(f"w{i}", 0, 1 - total)
                    total += weight
                    weights.append(weight)
                weights.append(1 - total)
                assert len(weights) == len(model_dir_paths)
                scores, _ = calc_all_scores(weights, post_process_modes=post_process_modes)
                return np.mean(scores)

            study = optuna.create_study(
                direction="maximize",
                storage="sqlite:///optuna-history.db",
                load_if_exists=True,
                study_name=models_dir_name,
            )
            study.enqueue_trial({f"w{i}": w for i, w in enumerate(weight_dict.values())})
            study.optimize(objective, n_trials=100, n_jobs=args.n_cpus, show_progress_bar=True)


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
