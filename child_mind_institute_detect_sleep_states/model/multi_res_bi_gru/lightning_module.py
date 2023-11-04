import copy
import warnings

import lightning as L
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.utils.data
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn

from ... import pj_struct_paths
from ...data.comp_dataset import get_event_df, get_submission_df
from ...score import calc_event_detection_ap
from ..dataset import UserWiseDataset
from .config import Config
from .model import MultiResidualBiGRU

__all__ = ["Module", "DataModule"]


class Module(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        self.optimizer_params = config["train"]["optimizer"]
        self.max_chunk_size = config["model"]["max_chunk_size"]
        self.model = MultiResidualBiGRU(
            input_size=2 * len(config["dataset"]["features"]),
            hidden_size=config["model"]["hidden_size"],
            out_size=config["model"]["out_size"],
            n_layers=config["model"]["n_layers"],
        )
        self.eval_time_window = config["eval"]["window"]
        self.step_interval = config["dataset"]["agg_interval"]
        self.train_dataset_type = config["dataset"]["train_dataset_type"]

        # self.early_stopping_epoch_ratio = (
        #     config["train"].get("early_stopping_max_epoch", config["train"]["num_epochs"])
        #     / config["train"]["num_epochs"]
        # )
        self.early_stopping_max_epoch = config["train"].get("early_stopping_max_epoch", config["train"]["num_epochs"])

        # self.preds_list: list = []
        # self.ids_list: list = []
        # self.steps_list: list = []
        self.val_data_list = []

        self.save_hyperparameters()

    def forward(self, batch):
        features, ids, *_ = batch

        pred = torch.zeros([*features.shape[:-1], 2]).to(self.device, non_blocking=True)

        h = None
        seq_len = features.shape[1]

        for i in range(0, seq_len, self.max_chunk_size):
            features_chunk = features[:, i : i + self.max_chunk_size].float().to(self.device, non_blocking=True)
            y_pred, h = self.model(features_chunk, h)
            h = [hi.detach() for hi in h]
            pred[:, i : i + self.max_chunk_size] = y_pred
        return pred

    def training_step(self, batch, batch_idx):
        _, _, _, y_batch = batch
        pred = self.forward(batch)

        loss = nn.MSELoss()(pred.float(), y_batch.float())
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, ids, steps, labels = batch

        preds = self.forward(batch)

        self.val_data_list.extend(
            [
                (id_.split("_")[0] if self.train_dataset_type == "with_part_id" else id_, step, pred)
                for id_, step, pred in zip(ids, steps.detach().cpu().numpy(), preds.detach().cpu().numpy(), strict=True)
            ]
        )

        # self.preds_list.append(pred.detach().cpu().numpy())
        # self.ids_list.append([id_.split("_")[0] for id_ in ids])
        # self.steps_list.append(steps.detach().cpu().numpy())

        # loss = nn.MSELoss()(preds.float(), labels.float())
        loss = nn.MSELoss()(preds, labels)
        self.log(
            "val_loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        # list_pred = [pred[0] for pred in self.preds_list]
        # series_ids = [ids[0] for ids in self.ids_list]
        # list_steps = self.steps_list

        assert len(self.val_data_list) > 0

        val_data_list = sorted(self.val_data_list, key=lambda row: row[0])  # sort by 'series_id'
        import itertools

        submission_df_list = []
        series_ids = []
        for series_id, grouped_iter in itertools.groupby(val_data_list, key=lambda row: row[0]):
            _, steps, preds = zip(*grouped_iter)
            series_ids.append(series_id)
            submission_df_list.append(
                get_submission_df(
                    np.concatenate(preds),
                    series_id,
                    np.concatenate(steps),
                    calc_type="top-probs",
                    step_interval=self.step_interval,
                    time_window=self.eval_time_window,
                )
            )
        submission_df = pd.concat(submission_df_list)

        event_df = get_event_df("train")
        event_df = event_df[event_df["series_id"].isin(np.unique(series_ids))][["series_id", "event", "step", "night"]]
        event_df = event_df.dropna()

        # submission_df = pd.concat(
        #     [
        #         get_submission_df(
        #             pred[np.newaxis],
        #             [uid],
        #             steps,
        #             calc_type="top-probs",
        #             step_interval=self.step_interval,
        #             time_window=self.eval_time_window,
        #         )
        #         for pred, uid, steps in zip(list_pred, series_ids, list_steps)
        #     ]
        # )

        score = calc_event_detection_ap(event_df, submission_df)
        self.log("EventDetectionAP", score, on_epoch=True, logger=True, prog_bar=True)

        # self.preds_list.clear()
        # self.ids_list.clear()
        # self.steps_list.clear()
        self.val_data_list.clear()

    def configure_optimizers(self):
        optimizer_params = copy.deepcopy(self.optimizer_params)

        scheduler_params = optimizer_params.pop("scheduler")
        # scheduler_params["T_max"] = self.trainer.estimated_stepping_batches * self.early_stopping_epoch_ratio
        scheduler_params["T_max"] = self.early_stopping_max_epoch

        print(f"{self.trainer.estimated_stepping_batches =}")
        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class DataModule(L.LightningDataModule):
    def __init__(self, df: pl.DataFrame | pl.LazyFrame, config: Config, i_fold: int):
        super().__init__()

        if isinstance(df, pl.DataFrame):
            n_total_records = df.select(pl.count())[0, 0]
        else:
            n_total_records = df.select(pl.count()).collect()[0, 0]

        self.df = df.with_columns(index=pl.Series(np.arange(n_total_records, dtype=np.uint32)))

        fold_dir_path = pj_struct_paths.get_data_dir_path() / "cmi-dss-train-k-fold-indices" / "base"
        self.fold_indices_npz_data = np.load(fold_dir_path / config["train"]["fold_type"] / f"fold{i_fold}.npz")
        self.config = config

        self.train_dataset = None
        self.valid_dataset = None

        self.target_series_ids = None
        if 0 < (config["train"]["lower_nan_fraction_to_exclude"] or -1) < 1:
            event_df = get_event_df("train")
            nan_fraction_df = event_df.groupby("series_id")["step"].apply(
                lambda steps: steps.isna().sum()
            ) / event_df.groupby("series_id")["step"].apply(len)
            self.target_series_ids = list(nan_fraction_df[nan_fraction_df < 0.8].index)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_df = self.df.filter(pl.col("index").is_in(np.where(self.fold_indices_npz_data["train"])[0])).drop(
                "index"
            )
            if self.target_series_ids is not None:
                print(f"nan ids removed: {len(train_df):,} -> ", end="")
                train_df = train_df.filter(pl.col("series_id").is_in(self.target_series_ids))
                print(f"{len(train_df):,}")
            # train_df = train_df.filter(
            #     pl.col("series_id").is_in(train_df.select(pl.col("series_id").unique().head(3)).collect()["series_id"])
            # )
            if self.config["dataset"]["train_dataset_type"] == "with_part_id":
                train_df = train_df.drop(columns=["series_id"]).drop_nulls("part_id").rename({"part_id": "series_id"})

            self.train_dataset = UserWiseDataset(
                train_df,
                agg_interval=self.config["dataset"]["agg_interval"],
                feature_names=self.config["dataset"]["features"],
                in_memory=self.config["dataset"].get("in_memory", True),
            )

        if stage in ("fit", "validate"):
            valid_df = self.df.filter(pl.col("index").is_in(np.where(self.fold_indices_npz_data["valid"])[0])).drop(
                "index"
            )
            self.valid_dataset = UserWiseDataset(
                valid_df,
                agg_interval=self.config["dataset"]["agg_interval"],
                feature_names=self.config["dataset"]["features"],
                in_memory=self.config["dataset"].get("in_memory", True),
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.config["train"]["train_batch_size"], shuffle=True, num_workers=8
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config["train"]["valid_batch_size"],
            shuffle=False,
            num_workers=8,
        )
