import multiprocessing
import os
import pathlib
from typing import Callable, Sequence

import cmi_dss_lib.utils.metrics
import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray

import child_mind_institute_detect_sleep_states.data.comp_dataset


def calc_score(
    i_fold: int,
    weights: list[float],
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


def get_grid(
    n_cols: int, step: float, target_sum: float = 1, start: float = 0
) -> NDArray[np.float_]:
    assert step < 1
    assert 0 <= target_sum

    target_sum *= round(1 / step)
    base_weight = pd.DataFrame(
        np.arange(round(1 / step) + 1) + round(start * round(1 / step)), dtype="i4"
    )

    weight = base_weight.copy()
    for i, _ in enumerate(tqdm.trange(n_cols - 1)):
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


def optimize(
    search_type: str,
    models_dir_name: str,
    calc_all_scores: Callable[
        [Sequence[float], dict | None, float, float], tuple[Sequence[float], Sequence[float]]
    ],
    weight,
    weight_dict,
    n_cpus=None,
):
    if n_cpus is None:
        n_cpus = os.cpu_count()

    print(f"{n_cpus = }")

    match search_type:
        case "grid_search":
            target_csv_path = (
                pathlib.Path(__file__).parent / "grid_search" / models_dir_name / "grid_search.csv"
            )
            print(f"{target_csv_path = }")

            if target_csv_path.exists():
                df = pd.read_csv(target_csv_path)
                df["scores"] = df["scores"].apply(
                    lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")]
                )
                df["weights"] = df["weights"].apply(
                    lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")]
                )

                # df = pd.concat(
                #     [
                #         df,
                #         pd.DataFrame(
                #             [(score_th, distance) for score_th, distance in df["weights"]],
                #             columns=["score_th", "distance"],
                #         ),
                #     ],
                #     axis=1,
                # ).sort_values(["score_th", "distance"])
                #
                # unique_score_ths = df["score_th"].unique()
                # unique_distances = df["distance"].unique()
                # grid = (
                #     pd.merge(
                #         pd.Series(unique_score_ths, name="score_th"),
                #         pd.Series(unique_distances, name="distance"),
                #         how="cross",
                #     )
                #     .to_numpy()
                #     .reshape((len(unique_score_ths), len(unique_distances), 2))
                # )
                #
                # indices = [
                #     np.argmax(
                #         np.isclose(score_th, grid[..., 0].flatten())
                #         & np.isclose(distance, grid[..., 1].flatten())
                #     )
                #     for score_th, distance in df[["score_th", "distance"]].values
                # ]
                # grid_score = np.full(grid.shape[:-1], np.nan).reshape(-1)
                # grid_score[indices] = df["CV"]
                # grid_score = grid_score.reshape(grid.shape[:-1])
                # import plotly.express as px
                # import plotly_utility.offline
                #
                # fig = px.imshow(
                #     grid_score,
                #     y=unique_score_ths,
                #     x=unique_distances,
                #     aspect=unique_distances.max() / unique_score_ths.max(),
                # )
                # fig = px.imshow(
                #     grid_score[:, :1],
                #     y=unique_score_ths,
                #     x=unique_distances[:1],
                #     aspect=unique_distances[:1].max() / unique_score_ths.max(),
                #     text_auto=".2f",
                # )
                # plotly_utility.offline.mpl_plot(fig)
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
            with multiprocessing.Pool(n_cpus) as p:
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
CV = {record_at_max["CV"]:.4f} ({", ".join(f"{s:.4f}" for s in record_at_max["scores"])})
weights = {dict(zip(weight_dict, record_at_max["weights"]))}
"""
            )
            return record_at_max
        case "optuna":
            import optuna

            def objective(trial: optuna.Trial):
                total = 0

                weights = []
                for i in range(len(weight_dict) - 1):
                    weight = trial.suggest_float(f"w{i}", 0, 1 - total)
                    total += weight
                    weights.append(weight)
                weights.append(1 - total)
                assert len(weights) == len(weight_dict)
                scores, _ = calc_all_scores(weights)
                return np.mean(scores)

            study = optuna.create_study(
                direction="maximize",
                storage="sqlite:///optuna-history.db",
                load_if_exists=True,
                study_name=models_dir_name,
            )
            study.enqueue_trial({f"w{i}": w for i, w in enumerate(weight_dict.values())})
            study.optimize(objective, n_trials=100, n_jobs=n_cpus, show_progress_bar=True)
