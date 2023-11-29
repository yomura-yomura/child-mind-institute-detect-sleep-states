import pathlib

import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.score.fast_event_detection_ap

project_root_path = pathlib.Path(__file__).parent.parent.parent

predicted_dir_path = (
    project_root_path
    / "run"
    / "predicted"
    # / "ranchantan/exp050-transformer-decoder_retry_resume/train/"
    / "jumtras/exp058/train/"
)


for i in range(5):
    print(f"fold {i + 1}")
    predicted_npz_paths = sorted((predicted_dir_path / f"fold_{i}").glob("*"))
    series_ids = [p.stem for p in predicted_npz_paths]

    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")
    event_df = event_df[event_df["series_id"].isin(series_ids)].dropna()

    sub_df_list = []
    for predicted_npz_path in predicted_npz_paths:
        series_id = predicted_npz_path.stem

        preds = np.load(predicted_npz_path)["arr_0"]
        sub_df_list.append(
            cmi_dss_lib.utils.post_process.post_process_for_seg(
                keys=[series_id] * len(preds),
                preds=preds,
                labels=["sleep", "event_onset", "event_wakeup"],
                downsample_rate=2,
                score_th=0.005,
                distance=96,
                post_process_modes=None,
            )
        )
    sub_df = pd.concat(sub_df_list)

    d = child_mind_institute_detect_sleep_states.score.fast_event_detection_ap.get_score_dict(event_df, sub_df)
    print(d)
    print()

    import plotly.express as px

    fig = px.imshow(
        np.stack([d["onset"], d["wakeup"]], axis=0),
        title=f"fold {i + 1}",
        x=list(map(str, child_mind_institute_detect_sleep_states.score.event_detection_ap.TOLERANCES)),
        y=["onset", "wakeup"],
        text_auto=".2f",
    )
    fig.show()
