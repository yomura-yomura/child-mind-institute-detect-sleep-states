import pandas as pd

from . import event_detection_ap


def calc_event_detection_ap(event_df: pd.DataFrame, submission_df: pd.DataFrame):
    if len(submission_df) == 0:
        ed_ap_score = 0
    else:
        ed_ap_score = event_detection_ap.score(
            event_df,
            submission_df,
            tolerances={k: [12, 36, 60, 90, 120, 150, 180, 240, 300, 360] for k in ["wakeup", "onset"]},
            series_id_column_name="series_id",
            time_column_name="step",
            event_column_name="event",
            score_column_name="score",
        )
    return ed_ap_score
