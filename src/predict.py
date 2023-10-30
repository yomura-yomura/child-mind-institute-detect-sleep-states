import argparse
import pathlib

import lightning.pytorch as lp
import numpy as np
import pandas as pd
import polars as pl
import toml

import child_mind_institute_detect_sleep_states as cmi_dss
import child_mind_institute_detect_sleep_states.model.sleep_stage_classification

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=pathlib.Path)
args = parser.parse_args(["config/multi_res_bi_gru.toml"])

with open(args.config_path) as f:
    config = toml.load(f)
    print(config)

# ckpt_path = sorted(pathlib.Path("models/test-#1-of-5/").glob("last-v*.ckpt"), key=lambda p: int(p.name[6:-5]))[-1]
# ckpt_path = "models/test-#1-of-5/last-v16.ckpt"
# ckpt_path = "models/test-#1-of-5/last-v38.ckpt"

import pathlib

import sklearn.model_selection
import torch.utils.data

import child_mind_institute_detect_sleep_states.model.multi_res_bi_gru

data_dir_path = pathlib.Path("data")
model_path = pathlib.Path("models")

# exp_name = "base"
# exp_name = "remove-nan"
# exp_name = "remove-0.3-nan"
exp_name = "remove-0.8-nan"


from tqdm.auto import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.score.event_detection_ap


def main(exp_name_dir_path):
    score_list = []

    for i_fold in range(config["train"]["n_folds"]):
        ckpt_dir_path = exp_name_dir_path / f"fold{i_fold + 1}"
        submission_path = ckpt_dir_path / "submission.csv"

        if submission_path.exists():
            submission_df = pd.read_csv(submission_path)
        else:
            trainer = lp.Trainer()

            module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.Module.load_from_checkpoint(
                ckpt_dir_path / "best.ckpt", cfg=config
            )

            fold_dir_path = data_dir_path / f"sigma{config['dataset']['sigma']}" / f"fold{i_fold}"
            p = fold_dir_path / "valid.parquet"
            valid_dataset = (
                child_mind_institute_detect_sleep_states.model.sleep_stage_classification.dataset.UserWiseDataset(
                    pl.scan_parquet(p),
                )
            )
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

            preds_list = trainer.predict(module, valid_loader)
            # preds = torch.concat(preds).numpy()

            submission_df_list = []
            prob_df_list = []
            for preds, batch in zip(preds_list, tqdm(valid_loader)):
                *_, uid, steps = batch

                submission_df_list.append(
                    cmi_dss.data.comp_dataset.get_submission_df(
                        preds.numpy(), uid, steps.numpy(), calc_type="top-probs"
                    )
                )
                prob_df_list.append(
                    pd.DataFrame(
                        {
                            "series_id": uid[0],
                            "step": steps[0],
                            "wakeup_prob": preds[0, :, 0],
                            "onset_prob": preds[0, :, 1],
                        }
                    )
                )

            prob_df = pd.concat(prob_df_list)
            assert len(prob_df) == len(prob_df[["series_id", "step"]].drop_duplicates())
            prob_df.to_csv(ckpt_dir_path / "prob.csv", index=False)

            submission_df = pd.concat(submission_df_list).sort_values(["series_id", "step", "event"])
            submission_df.to_csv(submission_path, index=False)

        event_df = cmi_dss.data.comp_dataset.get_event_df("train")
        event_df = event_df[event_df["series_id"].isin(submission_df["series_id"].unique())][
            ["series_id", "event", "step", "night"]
        ]
        event_df = event_df.dropna()
        score = cmi_dss.score.calc_event_detection_ap(
            # event_df, submission_df2[submission_df2["series_id"].isin(event_df["series_id"])]
            event_df,
            submission_df,
        )
        print(f"{score = :.3f}")
        score_list.append(score)

    mean_score_str, *scores_str = map("{:.3f}".format, (np.mean(score_list), *score_list))
    print(f"{mean_score_str} ({', '.join(scores_str)})")
    return score_list


# main(model_path / config["model_architecture"] / exp_name)

for p in (model_path / config["model_architecture"]).glob("*"):
    print(p)
    main(p)
