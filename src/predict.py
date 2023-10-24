import lightning.pytorch as lp
import numpy as np
import pandas as pd

import child_mind_institute_detect_sleep_states as cmi_dss
import child_mind_institute_detect_sleep_states.model.sleep_stage_classification

ckpt_path = "models/test-#1-of-5/last-v16.ckpt"

# steps_in_epoch = 6
# n_prev_time = 12 * 10
# steps_in_epoch = 12 * 30

prev_steps_in_epoch = 12 * 20
# next_steps_in_epoch = 12 * 10
next_steps_in_epoch = 1

n_prev_time = 1
# n_interval_steps = steps_in_epoch
n_interval_steps = prev_steps_in_epoch

# learning_rate = 0.001 * 64 / 16

device = "cuda"

module = child_mind_institute_detect_sleep_states.model.sleep_stage_classification.Module.load_from_checkpoint(
    ckpt_path,
    steps_in_epoch=prev_steps_in_epoch + next_steps_in_epoch,
)

df = pd.read_parquet("all-corrected.parquet")


n_folds = 5

import sklearn.model_selection
import torch.utils.data

kf = sklearn.model_selection.GroupKFold(n_splits=n_folds)
for i_fold, (_, valid_indices) in enumerate(kf.split(df, groups=df["series_id"])):
    for series_id, indices in df.iloc[valid_indices].groupby("series_id").indices.items():
        target_df = df.iloc[valid_indices].iloc[indices]
        valid_dataset = cmi_dss.model.sleep_stage_classification.get_dataset(
            target_df,
            device,
            prev_steps_in_epoch=prev_steps_in_epoch,
            next_steps_in_epoch=next_steps_in_epoch,
            n_prev_time=n_prev_time,
            n_interval_steps=n_interval_steps,
        )
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

        trainer = lp.Trainer()
        preds = trainer.predict(module, valid_loader)
        preds = torch.concat(preds).numpy()

        import polars as pl

        import child_mind_institute_detect_sleep_states.data.comp_dataset
        import child_mind_institute_detect_sleep_states.score.event_detection_ap

        submission_df1 = cmi_dss.data.comp_dataset.get_submission_df(
            preds, valid_dataset.uid, calc_type="max-along-type"
        )
        submission_df2 = cmi_dss.data.comp_dataset.get_submission_df(preds, valid_dataset.uid, calc_type="top-probs")

        event_df = cmi_dss.data.comp_dataset.get_event_df("train")
        event_df = event_df[event_df["series_id"] == series_id][["series_id", "event", "step", "night"]]
        event_df = event_df.dropna()
        score1 = cmi_dss.score.calc_event_detection_ap(event_df, submission_df1)
        print(f"{score1 = :.3f}")
        score2 = cmi_dss.score.calc_event_detection_ap(event_df, submission_df2)
        print(f"{score2 = :.3f}")
        break
    break
