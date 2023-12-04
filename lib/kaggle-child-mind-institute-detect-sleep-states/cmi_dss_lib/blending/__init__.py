import multiprocessing
import os
import pathlib
from typing import Callable, Sequence

import cmi_dss_lib.data.utils
import cmi_dss_lib.utils.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import tqdm
from nptyping import NDArray, Shape

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.score

project_root_path = pathlib.Path(__file__).parent.parent


start_timing_dict = cmi_dss_lib.data.utils.get_start_timing_dict("train")


def calc_score(
    i_fold: int,
    weights: list[float],
    keys_dict,
    all_event_df,
    preds_dict,
    post_process_modes,
    calc_type: str = "fast",
    score_th=0.005,
    distance=96,
    width=None,
    n_records_per_series_id=None,
    print_msg=False,
):
    series_ids = keys_dict[i_fold]
    # unique_series_ids = np.unique([str(k).split("_")[0] for k in keys])
    unique_series_ids = np.unique(series_ids)
    event_df = all_event_df[all_event_df["series_id"].isin(unique_series_ids)]

    df_submit_list = []
    for series_id, preds in zip(series_ids, preds_dict[i_fold], strict=True):
        assert preds.shape[0] == len(weights), (preds.shape, len(weights))
        mean_preds = np.average(preds, axis=0, weights=weights)

        # p = pathlib.Path("mean_preds") / "train" / f"fold_{i_fold}" / f"{series_id}.npz"
        # p.parent.mkdir(exist_ok=True, parents=True)
        # np.savez_compressed(p, mean_preds)
        # print(f"Info: saved as {p}")

        df = cmi_dss_lib.utils.post_process.post_process_for_seg(
            series_id=series_id,
            preds=mean_preds,
            labels=["sleep", "event_onset", "event_wakeup"],
            # preds=corrected_preds[:, :, [1, 2]],
            downsample_rate=2,
            score_th=score_th,
            distance=distance,
            width=width,
            post_process_modes=post_process_modes,
            print_msg=print_msg,
            n_records_per_series_id=n_records_per_series_id,
            start_timing_dict=start_timing_dict,
        )
        df_submit_list.append(df)
    df_submit = pd.concat(df_submit_list)
    df_submit = df_submit.sort_values(["series_id", "step"])

    if print_msg:
        print(df_submit.shape, len(df_submit) / len(series_ids))

    if calc_type == "fast":
        return child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(
            event_df[event_df["series_id"].isin(unique_series_ids)],
            df_submit,
            n_jobs=1,
            show_progress=False,
            print_score=print_msg,
        )
    else:
        return cmi_dss_lib.utils.metrics.event_detection_ap(
            event_df[event_df["series_id"].isin(unique_series_ids)], df_submit
        )


def get_grid(
    n_cols: int,
    step: float,
    target_sum: float = 1,
    start: float = 0,
    limits: list[tuple[float, float]] | None = None,
) -> NDArray[Shape["*, *"], np.float_]:
    assert step < 1
    assert 0 <= target_sum

    target_sum *= round(1 / step)
    base_weight = pd.DataFrame(np.arange(round(1 / step) + 1) + round(start * round(1 / step)), dtype="i4")

    weight = None
    for i, _ in enumerate(tqdm.trange(n_cols)):
        if weight is None:
            weight = base_weight.copy()
        else:
            weight = weight.merge(base_weight.rename(columns={0: i + 1}), how="cross")
        weight = weight[np.sum(weight, axis=1) <= target_sum]
        if limits is not None:
            min_w, max_w = limits[i]
            weight = weight[
                (np.floor(min_w / step) <= weight.iloc[:, -1]) & (weight.iloc[:, -1] <= np.ceil(max_w / step))
            ]
        weight = weight.reset_index(drop=True)
    weight = weight.to_numpy()
    weight = weight[np.sum(weight, axis=1) == target_sum]
    print(f"{weight.shape = }")
    return weight * step


def get_keys_and_preds(model_dir_paths: list[pathlib.Path | str], folds: list[int]):
    predicted_npz_dir_paths = [
        [pathlib.Path(model_dir_path) / "train" / f"fold_{i_fold}" for model_dir_path in model_dir_paths]
        for i_fold in folds
    ]  # (fold, model)
    for predicted_npz_dir_paths_by_fold in predicted_npz_dir_paths:
        for path in predicted_npz_dir_paths_by_fold:
            if not path.exists():
                raise FileNotFoundError(path)

    count_by_series_id_df = (
        child_mind_institute_detect_sleep_states.data.comp_dataset.get_series_df("train", as_polars=True)
        .group_by("series_id")
        .count()
        .collect()
    )
    min_duration_dict = dict(count_by_series_id_df.iter_rows())

    keys_dict = {}
    preds_dict = {}
    for i, i_fold in enumerate(tqdm.tqdm(folds)):
        (
            keys_dict[i_fold],
            preds_dict[i_fold],
        ) = cmi_dss_lib.utils.common.load_predicted_npz_group_by_series_id(
            predicted_npz_dir_paths[i], min_duration_dict
        )
    return keys_dict, preds_dict


def optimize(
    search_type: str,
    models_dir_name: str,
    calc_all_scores: Callable[
        [list[float], dict | None, float, float, str, int, bool],
        tuple[Sequence[float], Sequence[float]],
    ],
    all_weights_to_find: NDArray | None,
    limits: list[tuple[float, float]] | None,
    initial_weight_dict=None,
    n_cpus=None,
    objective=None,
):
    if n_cpus is None:
        n_cpus = os.cpu_count()

    print(f"{n_cpus = }")

    match search_type:
        case "grid_search":
            target_csv_path = project_root_path / "run" / "grid_search" / models_dir_name / "grid_search.csv"
            print(f"{target_csv_path = }")

            if target_csv_path.exists():
                df = pd.read_csv(target_csv_path)
                df["scores"] = df["scores"].apply(lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")])
                df["weights"] = df["weights"].apply(lambda w: [float(n.strip("' ")) for n in w.strip("[]").split(",")])

                loaded_weight = np.array(df["weights"].tolist())
                all_weights_to_find = np.array(
                    [w for w in all_weights_to_find if not np.any(np.all(np.isclose(w, loaded_weight), axis=1))]
                )
                print(f"-> {all_weights_to_find.shape = }")

                records = df.to_dict("records")
            else:
                records = []

            n_steps_to_save = 30
            with multiprocessing.Pool(n_cpus) as p:
                with tqdm.tqdm(total=len(all_weights_to_find), desc="grid search") as t:
                    for scores, weights in p.imap_unordered(calc_all_scores, all_weights_to_find.tolist()):
                        t.update(1)
                        records.append({"CV": np.mean(scores), "scores": scores, "weights": weights})

                        if len(records) % n_steps_to_save == 0:
                            print_best_score_so_far(records, target_csv_path)

            record_at_max = print_best_score_so_far(records, target_csv_path)
            return record_at_max
        case "optuna":
            assert all_weights_to_find is None

            import optuna

            from .optuna_blending import run

            db_path = pathlib.Path("optuna-history") / f"{models_dir_name}.db"
            db_path.parent.mkdir(exist_ok=True, parents=True)
            study = optuna.create_study(
                direction="maximize",
                storage=f"sqlite:///{db_path}",
                load_if_exists=True,
                study_name=models_dir_name,
            )
            study.enqueue_trial({f"w{i}": w for i, w in enumerate(initial_weight_dict.values())})

            try:
                print(f"{study.best_value = :.4f}")
                print(f"{study.best_params = }")
            except ValueError:
                pass

            with multiprocessing.Pool(n_cpus) as p:
                for _ret in p.starmap(run, [(i_cpu, models_dir_name, db_path, objective) for i_cpu in range(n_cpus)]):
                    pass


def print_best_score_so_far(records: list[dict], target_csv_path: pathlib.Path, initial_weight_dict=None):
    df = pd.DataFrame(records)
    target_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_csv_path, index=False)

    record_at_max = df.iloc[df["CV"].argmax()]

    if initial_weight_dict is None:
        weights = record_at_max["weights"]
    else:
        weights = dict(zip(initial_weight_dict, record_at_max["weights"]))
    print(
        f"""
    max:
    CV = {record_at_max["CV"]:.4f}
    weights = {weights}
    """
    )
    return record_at_max
