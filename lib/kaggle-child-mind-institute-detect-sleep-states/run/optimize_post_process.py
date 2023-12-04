import pathlib

import numpy as np
import tqdm
from cmi_dss_lib.blending import calc_score, get_keys_and_preds, optimize

import child_mind_institute_detect_sleep_states.data.comp_dataset

project_root_path = pathlib.Path(__file__).parent.parent
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"
train_pred_dir_path = project_root_path / "run" / "predicted" / "train"
blending_pred_dir_path = project_root_path / "run" / "predicted" / "blending"

# model_dir_path = ranchantan_pred_dir_path / "exp050-transformer-decoder_retry"

# model_dir_path = train_pred_dir_path / "exp104_2"
# folds = [0]

model_dir_path = blending_pred_dir_path / "exp033"
folds = [0, 1, 2, 3, 4]

# model_dir_path = blending_pred_dir_path / "exp026"
# folds = [0, 1, 2, 3, 4]

if __name__ == "__main__":
    # post_process_type = "base"
    post_process_type = "cutting_probs_by_sleep_prob"
    # post_process_type = "sleeping_edges_as_probs"

    keys_dict, preds_dict = get_keys_and_preds([model_dir_path], folds=folds)

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    import pandas as pd

    if post_process_type == "base":
        score_ths = np.logspace(-5, 1, 100)
        # distances = np.linspace(1, 1 * 12 * 60, 100)
        distances = np.linspace(1, 100, 100)
        # score_ths = np.logspace(-5, np.log10(0.1), 10)[::-1]
        # distances = np.linspace(1, 100, 101)
        # score_ths = [1e-5]
        # distances = [90]
        # widths = np.linspace(0, 300, 30)
        widths = [0]
        # score_ths = np.logspace(-3, 0, 10)
        # distances = np.linspace(200, 300, 10)
        base_parameters = [
            pd.Series(score_ths, name="score_th"),
            pd.Series(distances, name="distance"),
            pd.Series(widths, name="width"),
        ]
        grid_parameters = None
        for param in base_parameters:
            if grid_parameters is None:
                grid_parameters = param
            else:
                grid_parameters = pd.merge(
                    grid_parameters,
                    param,
                    how="cross",
                )
        grid_parameters = grid_parameters.to_numpy()
        print(f"{grid_parameters = }")

        def calc_all_scores(grid_parameter, calc_type="fast"):
            score_th, distance, width = grid_parameter
            scores = [
                calc_score(
                    i_fold,
                    [1],
                    keys_dict,
                    all_event_df,
                    preds_dict,
                    None,
                    score_th=score_th,
                    distance=distance,
                    width=width,
                    calc_type=calc_type,
                    n_records_per_series_id=2000,
                )
                for i_fold in tqdm.tqdm(folds, desc="calc score over n-folds")
            ]
            mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
            print(f"{mean_score_str} ({', '.join(score_strs)}) at {grid_parameter}")
            return scores, grid_parameter

    elif post_process_type == "cutting_probs_by_sleep_prob":
        # sleep_occupancy_ths = np.linspace(0, 0.1, 10 + 1)
        # watch_interval_hours = np.linspace(0, 12, 24 + 1)
        sleep_occupancy_ths = np.linspace(0, 1, 10 + 1)
        watch_interval_hours = np.arange(6, 9, 0.5)
        n_continuous_list = np.arange(0, 9, 1)
        # grid_parameters = pd.merge(
        #     pd.merge(
        #         pd.Series(sleep_occupancy_ths, name="sleep_occupancy_th_onset"),
        #         pd.Series(watch_interval_hours, name="watch_interval_hour_onset"),
        #         how="cross",
        #     ),
        #     pd.merge(
        #         pd.Series(sleep_occupancy_ths, name="sleep_occupancy_th_wakeup"),
        #         pd.Series(watch_interval_hours, name="watch_interval_hour_wakeup"),
        #         how="cross",
        #     ),
        #     how="cross",
        # ).to_numpy()
        grid_parameters = pd.merge(
            pd.merge(
                pd.Series(sleep_occupancy_ths, name="sleep_occupancy_th_onset"),
                pd.Series(watch_interval_hours, name="watch_interval_hour_onset"),
                how="cross",
            ),
            pd.Series(n_continuous_list, name="n_continuous"),
            how="cross",
        )

        def calc_all_scores(
            grid_parameter, score_th: float = 0.0005, distance: int = 96, calc_type: str = "fast"
        ):
            sleep_occupancy_th, watch_interval_hour, n_continuous = grid_parameter
            # (
            #     sleep_occupancy_th_onset,
            #     watch_interval_hour_onset,
            #     sleep_occupancy_th_wakeup,
            #     watch_interval_hour_wakeup,
            # ) = grid_parameter

            scores = [
                calc_score(
                    i_fold,
                    [1],
                    keys_dict,
                    all_event_df,
                    preds_dict,
                    score_th=score_th,
                    distance=distance,
                    calc_type=calc_type,
                    post_process_modes=dict(
                        cutting_probs_by_sleep_prob=dict(
                            sleep_occupancy_th=sleep_occupancy_th,
                            watch_interval_hour=watch_interval_hour,
                            n_continuous=n_continuous,
                            version=1
                            # onset=dict(
                            #     sleep_occupancy_th=sleep_occupancy_th_onset,
                            #     watch_interval_hour=watch_interval_hour_onset,
                            # ),
                            # wakeup=dict(
                            #     sleep_occupancy_th=sleep_occupancy_th_wakeup,
                            #     watch_interval_hour=watch_interval_hour_wakeup,
                            # ),
                        ),
                    ),
                    print_msg=False,
                    n_records_per_series_id=2000,
                )
                for i_fold in tqdm.tqdm(folds, desc="calc score over n-folds")
            ]
            mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
            print(f"{mean_score_str} ({', '.join(score_strs)}) at {grid_parameter}")
            return scores, grid_parameter

    elif post_process_type == "sleeping_edges_as_probs":
        sleep_prob_ths = np.linspace(0, 1, 20 + 1)
        min_sleeping_hours = np.linspace(0, 12, 24 + 1)
        grid_parameters = pd.merge(
            pd.Series(sleep_prob_ths, name="sleep_prob_th"),
            pd.Series(min_sleeping_hours, name="min_sleeping_hour"),
            how="cross",
        ).to_numpy()

        def calc_all_scores(
            grid_parameter, score_th: float = 0.0005, distance: int = 96, calc_type: str = "fast"
        ):
            sleep_prob_th, min_sleeping_hours = grid_parameter

            scores = [
                calc_score(
                    i_fold,
                    [1],
                    keys_dict,
                    all_event_df,
                    preds_dict,
                    score_th=score_th,
                    distance=distance,
                    calc_type=calc_type,
                    post_process_modes=dict(
                        sleeping_edges_as_probs=dict(
                            sleep_prob_th=sleep_prob_th, min_sleeping_hours=min_sleeping_hours
                        )
                    ),
                    print_msg=False,
                    n_records_per_series_id=2000,
                )
                for i_fold in tqdm.tqdm(folds, desc="calc score over n-folds")
            ]
            mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
            print(f"{mean_score_str} ({', '.join(score_strs)}) at {grid_parameter}")
            return scores, grid_parameter

    else:
        raise ValueError(f"unexpected {post_process_type=}")

    record_at_max = optimize(
        "grid_search",
        f"post_process/{post_process_type}-v3/{model_dir_path.name}",
        calc_all_scores,
        grid_parameters,
        limits=None,
    )
    calc_all_scores(record_at_max["weights"], calc_type="normal")
