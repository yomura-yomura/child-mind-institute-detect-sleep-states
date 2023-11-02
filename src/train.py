import argparse
import pathlib

import lightning as L
import polars as pl
import toml
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from child_mind_institute_detect_sleep_states.model.callbacks import ModelCheckpointWithSymlinkToBest
from child_mind_institute_detect_sleep_states.model.loggers import WandbLogger, get_versioning_wandb_group_name

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=pathlib.Path)
parser.add_argument("--n-devices", "-n", type=int, default=1)
parser.add_argument("-f", default=False, action="store_true")
args = parser.parse_args(
    # [
    #     # "config/sleep_stage_classification.toml"
    #     "config/multi_res_bi_gru.toml",
    #     "-f",
    # ]
)

with open(args.config_path) as f:
    config = toml.load(f)
    print(config)

data_dir_path = pathlib.Path("data") / "base"


exp_name = config["exp_name"]

wandb_group_name = f"{config['model_architecture']}-{exp_name}"
wandb_group_name = get_versioning_wandb_group_name(wandb_group_name)
# wandb_group_name = f"{wandb_group_name}_v6"

model_path = pathlib.Path("models")

exp_name_dir_path = model_path / config["model_architecture"] / config["train"]["fold_type"] / exp_name

config_path = exp_name_dir_path / "config.toml"
if not args.f and config_path.exists():
    raise FileExistsError(config_path)


config_path.parent.mkdir(exist_ok=True, parents=True)
with open(config_path, "w") as f:
    toml.dump(config, f)

fold_dir_path = pathlib.Path("cmi-dss-train-k-fold-indices") / "base"


df = pl.scan_parquet(data_dir_path / f"all-corrected-sigma{config['dataset']['sigma']}.parquet")


for i_fold in range(config["train"]["n_folds"]):
    # if i_fold <= 0:
    #     continue
    if (exp_name_dir_path / f"fold{i_fold + 1}").exists():
        continue

    name = "-".join(
        [
            config["model_architecture"],
            config["train"]["fold_type"],
            exp_name,
            f"#{i_fold + 1}-of-{config['train']['n_folds']}",
        ]
    )
    print(name)

    # print(
    #     pl.scan_parquet(p)
    #     .select(pl.col("series_id").unique().is_in(target_series_ids))
    #     .collect()
    #     .to_pandas()
    #     .value_counts()
    # )

    match config["model_architecture"]:
        case "multi_res_bi_gru":
            import child_mind_institute_detect_sleep_states.model.multi_res_bi_gru

            data_module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.DataModule(df, config, i_fold)

            # config["train"]["optimizer"]["scheduler"]["T_max"] = 20 * len(data_module.train_dataset)
            module = child_mind_institute_detect_sleep_states.model.multi_res_bi_gru.Module(config)
            
        case "multi_res_bi_lstm":
            import child_mind_institute_detect_sleep_states.model.multi_res_bi_lstm

            #config["train"]["optimizer"]["scheduler"]["T_max"] = 20 * len(train_loader)
            module = child_mind_institute_detect_sleep_states.model.multi_res_bi_lstm.Module(config)
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
                dirpath=exp_name_dir_path / f"fold{i_fold + 1}",
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
    trainer.fit(module, data_module)

    wandb.finish()

    # break

# preds = trainer.predict(module, valid_loader)
# preds = torch.concat(preds)
# print(f"{preds = }")
# print(f"{preds.numpy().max(axis=0) = }")
