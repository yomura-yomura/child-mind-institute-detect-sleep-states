import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import tqdm
from cmi_dss_lib.blending import calc_score, get_keys_and_preds, optimize

import child_mind_institute_detect_sleep_states.data.comp_dataset

project_root_path = pathlib.Path(__file__).parent.parent
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"

model_dir_path = ranchantan_pred_dir_path / "exp050-transformer-decoder_retry"


if __name__ == "__main__":
    # post_process_type = "base"
    post_process_type = "cutting_probs_by_sleep_prob"

    keys_dict, preds_dict = get_keys_and_preds([model_dir_path])

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    import pandas as pd

    if post_process_type == "base":
        # score_ths = np.logspace(-3, 1, 100)
        # distances = np.linspace(1, 1 * 12 * 60, 100)
        score_ths = np.logspace(-10, np.log10(0.05), 100)[::-1]
        distances = np.linspace(1, 100, 101)
        grid_parameters = pd.merge(
            pd.Series(score_ths, name="score_th"),
            pd.Series(distances, name="distance"),
            how="cross",
        ).to_numpy()

        def calc_all_scores(grid_parameter, calc_type="fast"):
            score_th, distance = grid_parameter
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
                    calc_type=calc_type,
                )
                for i_fold in tqdm.trange(5, desc="calc score over n-folds")
            ]
            mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
            print(f"{mean_score_str} ({', '.join(score_strs)}) at {grid_parameter}")
            return scores, grid_parameter

    elif post_process_type == "cutting_probs_by_sleep_prob":
        grid_parameters = pd.merge(
            pd.Series(np.linspace(0, 0.1, 10 + 1), name="sleep_occupancy_th"),
            pd.Series(np.linspace(0, 12, 24 + 1), name="watch_interval_hour"),
            how="cross",
        ).to_numpy()

        def calc_all_scores(grid_parameter, score_th=0.0005, distance=96, calc_type="fast"):
            sleep_occupancy_th, watch_interval_hour = grid_parameter
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
                        )
                    ),
                    print_msg=False,
                )
                for i_fold in tqdm.trange(5, desc="calc score over n-folds")
            ]
            mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
            print(f"{mean_score_str} ({', '.join(score_strs)}) at {grid_parameter}")
            return scores, grid_parameter

    else:
        raise ValueError(f"unexpected {post_process_type=}")

    record_at_max = optimize(
        "grid_search",
        f"post_process/{post_process_type}/{model_dir_path.name}",
        calc_all_scores,
        grid_parameters,
    )
    calc_all_scores(record_at_max["weights"], calc_type="normal")
