import argparse
import pathlib

import lightning as L
import polars as pl
import toml
import torch.utils.data
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import child_mind_institute_detect_sleep_states.model.sleep_stage_classification

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=pathlib.Path)
parser.add_argument("--n-devices", "-n", type=int, default=1)
args = parser.parse_args(
    [
        # "config/sleep_stage_classification.toml"
        "config/multi_res_bi_gru.toml"
    ]
)

with open(args.config_path) as f:
    config = toml.load(f)
    print(config)

data_dir_path = pathlib.Path("data")


# exp_name = "remove-0.3-nan"
# exp_name = "remove-0.8-nan"
exp_name = "remove-0.8-nan-interval-6"

wandb_group_name = f"{config['model_architecture']}-{exp_name}"

api = wandb.Api()
possible_exp_names = set(
    run._attrs["group"]
    for run in api.runs(
        "ranchan/child-mind-institute-detect-sleep-states",
        filters={"group": {"$regex": rf"^{wandb_group_name}(_v\d+)?$"}},
    )
)

if len(possible_exp_names) == 0:
    target_version = 0
else:
    target_version = 1

possible_exp_names -= {wandb_group_name}
if len(possible_exp_names) > 0:
    target_version += max(int(name[len(f"{wandb_group_name}_v") :]) for name in possible_exp_names)

if target_version > 0:
    wandb_group_name = f"{wandb_group_name}_v{target_version}"


print(f"{wandb_group_name = }")


for i_fold in range(config["train"]["n_folds"]):
    # if i_fold <= 0:
    #     continue

    fold_dir_path = data_dir_path / f"sigma{config['dataset']['sigma']}" / f"fold{i_fold}"

    name = f"{config['model_architecture']}-{exp_name}-#{i_fold + 1}-of-{config['train']['n_folds']}"
    print(name)

    import child_mind_institute_detect_sleep_states.data

    event_df = child_mind_institute_detect_sleep_states.data.comp_dataset.get_event_df("train")
    nan_fraction_df = event_df.groupby("series_id")["step"].apply(lambda steps: steps.isna().sum()) / event_df.groupby(
        "series_id"
    )["step"].apply(len)
    target_series_ids = list(nan_fraction_df[nan_fraction_df < 0.8].index)

    p = fold_dir_path / "train.parquet"
    train_dataset = child_mind_institute_detect_sleep_states.model.sleep_stage_classification.dataset.UserWiseDataset(
        # pl.scan_parquet(p).filter(
        #     pl.col("series_id").is_in(
        #         list(pl.scan_parquet(p).select(pl.col("series_id").unique()).head(3).collect()["series_id"])
        #     )
        # )
        pl.scan_parquet(p).filter(pl.col("series_id").is_in(target_series_ids)),
        agg_interval=config["dataset"]["agg_interval"],
        feature_names=config["dataset"]["features"],
    )

    p = fold_dir_path / "valid.parquet"
    valid_dataset = child_mind_institute_detect_sleep_states.model.sleep_stage_classification.dataset.UserWiseDataset(
        # pl.scan_parquet(p).filter(
        #     pl.col("series_id").is_in(
        #         list(pl.scan_parquet(p).select(pl.col("series_id").unique()).head(3).collect()["series_id"])
        #     )
        # )
        pl.scan_parquet(p),
        agg_interval=config["dataset"]["agg_interval"],
        feature_names=config["dataset"]["features"],
    )
    # print(
    #     pl.scan_parquet(p)
    #     .select(pl.col("series_id").unique().is_in(target_series_ids))
    #     .collect()
    #     .to_pandas()
    #     .value_counts()
    # )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config["train"]["train_batch_size"], shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=config["train"]["valid_batch_size"], shuffle=False
    )

    match config["model_architecture"]:
        case "multi_res_bi_gru":
            import child_mind_institute_detect_sleep_states.model.multi_res_bi_gru

            config["train"]["optimizer"]["scheduler"]["T_max"] = 20 * len(train_loader)
            module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.Module(config)
        case "sleep_stage_classification":
            module = child_mind_institute_detect_sleep_states.model.sleep_stage_classification.Module(
                # n_features=2,
                n_features=10,
                learning_rate=config["train"]["learning_rate"],
                wakeup_weight=config["train"]["weight"],
                onset_weight=config["train"]["weight"],
            )
        case _ as model_architecture:
            raise ValueError(f"{model_architecture=} not expected")

    model_path = pathlib.Path("models")

    from torch import Tensor

    class ModelCheckpointWithSymlinkToBest(ModelCheckpoint):
        CHECKPOINT_NAME_BEST = "best"

        def _save_last_checkpoint(self, trainer: "L.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
            """
            save last+best checkpoint
            """

            # save last
            super()._save_last_checkpoint(trainer, monitor_candidates)

            # save best below
            filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_BEST)

            if self._enable_version_counter:
                version_cnt = self.STARTING_VERSION
                while self.file_exists(filepath, trainer) and filepath != getattr(self, "previous_best_model_path", ""):
                    filepath = self.format_checkpoint_name(
                        monitor_candidates, self.CHECKPOINT_NAME_BEST, ver=version_cnt
                    )
                    version_cnt += 1

            # set the last model path before saving because it will be part of the state.
            previous, self.previous_best_model_path = getattr(self, "previous_best_model_path", ""), filepath
            if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
                self._link_checkpoint(trainer, self.best_model_path, filepath)
            else:
                self._save_checkpoint(trainer, filepath)
            if previous and self._should_remove_checkpoint(trainer, previous, filepath):
                self._remove_checkpoint(trainer, previous)

    trainer = L.Trainer(
        devices=args.n_devices,
        max_epochs=config["train"]["num_epochs"],
        logger=WandbLogger(
            project="child-mind-institute-detect-sleep-states", name=name, save_dir="wandb_logs", group=wandb_group_name
        ),
        callbacks=[
            LearningRateMonitor(),
            EarlyStopping(
                monitor="EventDetectionAP",
                mode="max",
                patience=config["train"]["early_stopping_patience"],
            ),
            ModelCheckpointWithSymlinkToBest(
                dirpath=model_path / config["model_architecture"] / exp_name / f"fold{i_fold + 1}",
                filename="{epoch}-{EventDetectionAP:.3f}",
                monitor="EventDetectionAP",
                mode="max",
                save_last=True,
                save_top_k=config["train"]["save_top_k_models"],
                # every_n_train_steps=eval_steps,
            ),
        ],
        # val_check_interval=eval_steps,
    )

    print("fitting")
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()

    # break

# preds = trainer.predict(module, valid_loader)
# preds = torch.concat(preds)
# print(f"{preds = }")
# print(f"{preds.numpy().max(axis=0) = }")
