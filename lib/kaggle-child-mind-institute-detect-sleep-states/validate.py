import pathlib

import pandas as pd
from cmi_dss_lib.utils.metrics import event_detection_ap, tolerances
from omegaconf import OmegaConf

import child_mind_institute_detect_sleep_states.data.comp_dataset

submission_df = pd.read_csv(
    "submission.csv"
    # "new_submission2.csv"
    # "new_submission3.csv"
)

# ids_dict = OmegaConf.load(pathlib.Path("run") / "conf" / "split" / "fold_0.yaml")
# submission_df = submission_df[submission_df["series_id"].isin(ids_dict["valid_series_ids"])]
submission_df = (
    submission_df.sort_values(["series_id", "step", "score"], ascending=[True, True, False])
    .groupby(["series_id", "step"])
    .head(1)
)

event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")

# event_df = event_df[event_df["series_id"].isin(ids_dict["valid_series_ids"])]
event_df = event_df[event_df["series_id"].isin(submission_df["series_id"])]
# score = child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(
#     event_df, submission_df
# )

score = event_detection_ap(
    solution=event_df,
    submission=submission_df,
    # tolerances=tolerances
    # submission_df[submission_df["score"] > 0.1],
)
print(f"{score = :.3f}")
