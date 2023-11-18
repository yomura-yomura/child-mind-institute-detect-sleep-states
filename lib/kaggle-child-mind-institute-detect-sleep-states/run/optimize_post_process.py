import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import tqdm
from blending import all_model_dir_path_dict, calc_score, get_keys_and_preds, optimize

import child_mind_institute_detect_sleep_states.data.comp_dataset

project_root_path = pathlib.Path(__file__).parent.parent
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"

model_dir_path = ranchantan_pred_dir_path / "exp050-transformer-decoder_retry"


if __name__ == "__main__":
    keys_dict, preds_dict = get_keys_and_preds(
        [model_dir_path]
        # list(all_model_dir_path_dict.values())
    )

    # stacked_preds = preds_dict[0][0]
    #
    # i = 20
    # duration = 24 * 12 * 60
    # preds = stacked_preds[:, i * duration : (i + 1) * duration]
    #
    # import plotly.express as px
    #
    # fig = px.imshow(preds[..., 2], aspect=preds.shape[0] / preds.shape[1])
    # fig.show()

    all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
        "train"
    ).dropna()

    # score_th = 0.005
    # distance = 96

    import pandas as pd

    # score_ths = np.logspace(-3, 1, 100)
    # distances = np.linspace(1, 1 * 12 * 60, 100)
    score_ths = np.logspace(-10, np.log10(0.05), 100)[::-1]
    distances = np.linspace(1, 100, 101)
    grid_parameters = pd.merge(
        pd.Series(score_ths, name="score_th"), pd.Series(distances, name="distance"), how="cross"
    ).to_numpy()

    def calc_all_scores(grid_parameter):
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
            )
            for i_fold in tqdm.trange(5, desc="calc score over n-folds")
        ]
        mean_score_str, *score_strs = map("{:.3f}".format, [np.mean(scores), *scores])
        print(f"{mean_score_str} ({', '.join(score_strs)}) at {grid_parameter}")
        return scores, grid_parameter

    optimize(
        "grid_search", f"post_process/{model_dir_path.name}", calc_all_scores, grid_parameters
    )
