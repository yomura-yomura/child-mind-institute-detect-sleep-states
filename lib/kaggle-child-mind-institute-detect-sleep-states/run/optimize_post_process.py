import pathlib

import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import tqdm
from blending import all_model_dir_path_dict, calc_score, get_keys_and_preds

import child_mind_institute_detect_sleep_states.data.comp_dataset

project_root_path = pathlib.Path(__file__).parent.parent
ranchantan_pred_dir_path = project_root_path / "run" / "predicted" / "ranchantan"

keys_dict, preds_dict = get_keys_and_preds(
    # [ranchantan_pred_dir_path / "exp041"]
    list(all_model_dir_path_dict.values())
)

stacked_preds = preds_dict[0][0]

i = 20
duration = 24 * 12 * 60
preds = stacked_preds[:, i * duration: (i + 1) * duration]

import plotly.express as px

fig = px.imshow(preds[..., 2], aspect=preds.shape[0] / preds.shape[1])
fig.show()

fdsa

all_event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df(
    "train"
).dropna()

score_th = 0.005
distance = 96
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
    for i_fold in tqdm.trange(5)
]
