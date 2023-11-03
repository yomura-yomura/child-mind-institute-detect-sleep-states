import argparse
import pathlib

import lightning.pytorch as lp
import numpy as np
import pandas as pd
import polars as pl
import toml
from tqdm.auto import tqdm

import child_mind_institute_detect_sleep_states.data.comp_dataset
import child_mind_institute_detect_sleep_states.model.multi_res_bi_gru
import child_mind_institute_detect_sleep_states.model.sleep_stage_classification
import child_mind_institute_detect_sleep_states.score.event_detection_ap

data_dir_path = pathlib.Path("data")
model_path = pathlib.Path("models")


def main(exp_name_dir_path: pathlib.Path, recreate: bool = False):
    score_list = []

    with open(exp_name_dir_path / "config.toml") as f:
        config = toml.load(f)
        print(config)

    df = pl.scan_parquet(data_dir_path / "base" / f"all-corrected-sigma{config['dataset']['sigma']}.parquet")

    for i_fold in range(config["train"]["n_folds"]):
        ckpt_dir_path = exp_name_dir_path / f"fold{i_fold + 1}"
        submission_path = ckpt_dir_path / "submission.csv"

        if not recreate and submission_path.exists():
            submission_df = pd.read_csv(submission_path)
        else:
            trainer = lp.Trainer()

            print(f"load from {ckpt_dir_path}")
            module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.Module.load_from_checkpoint(
                ckpt_dir_path / "best.ckpt", config=config
            )

            data_module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.DataModule(df, config, i_fold)
            data_module.setup("validate")

            preds_list = trainer.predict(module, data_module.val_dataloader())

            submission_df_list = []
            prob_df_list = []
            for preds, batch in zip(preds_list, tqdm(data_module.val_dataloader())):
                _, uid, steps, *_ = batch

                submission_df_list.append(
                    child_mind_institute_detect_sleep_states.data.comp_dataset.get_submission_df(
                        preds.numpy(),
                        uid,
                        steps.numpy(),
                        calc_type="top-probs",
                        step_interval=config["dataset"]["agg_interval"],
                        time_window=config["eval"]["window"],
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

        event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")
        event_df = event_df[event_df["series_id"].isin(submission_df["series_id"].unique())][
            ["series_id", "event", "step", "night"]
        ]
        event_df = event_df.dropna()
        score = child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(
            # event_df, submission_df2[submission_df2["series_id"].isin(event_df["series_id"])]
            event_df,
            submission_df,
        )
        print(f"#{i_fold + 1} of {config['train']['n_folds']}: {score = :.3f}")
        score_list.append(score)

    mean_score_str, *scores_str = map("{:.3f}".format, (np.mean(score_list), *score_list))
    print(f"{mean_score_str} ({', '.join(scores_str)})")
    return score_list


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_path", type=pathlib.Path)
    # args = parser.parse_args(
    #     ["config/multi_res_bi_gru.toml"]
    # )
    # main(model_path / "multi_res_bi_gru" / "group" / "remove-0.8-nan", recreate=True)
    # main(model_path / "multi_res_bi_gru" / "0.8-nan-3-interval", recreate=True)
    # main(model_path / "multi_res_bi_gru" / "stratified_group" / "0.8-nan-12-interval-with-fft", recreate=True)
    main(model_path / "multi_res_bi_gru" / "base" / "group" / "0.8-nan-12-interval-patient-10")
    # main(model_path / "multi_res_bi_gru" / "stratified_group" / "0.8-nan-12-interval", recreate=True)

    # for p in (model_path / "multi_res_bi_gru").glob("*"):
    #     print(p)
    #     main(p, recreate=True)
