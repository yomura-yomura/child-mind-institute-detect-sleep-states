from typing import Literal

import pandas as pd

from . import event_detection_ap, fast_event_detection_ap


def calc_event_detection_ap(
    event_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    calc_type: Literal["normal", "fast"] = "fast",
    n_jobs: int = -1,
    show_progress: bool = True,
    print_score: bool = True,
):
    if len(submission_df) == 0:
        return

    if calc_type == "normal":
        return event_detection_ap.score(
            event_df,
            submission_df,
            tolerances={k: event_detection_ap.TOLERANCES for k in ["wakeup", "onset"]},
            series_id_column_name="series_id",
            time_column_name="step",
            event_column_name="event",
            score_column_name="score",
        )
    elif calc_type == "fast":
        return fast_event_detection_ap.score(
            event_df, submission_df, n_jobs=n_jobs, show_progress=show_progress, print_score=print_score
        )
