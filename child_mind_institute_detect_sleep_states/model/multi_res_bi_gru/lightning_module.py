import copy

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import nn

from ...data.comp_dataset import get_event_df, get_submission_df
from ...score import calc_event_detection_ap
from .model import MultiResidualBiGRU

__all__ = ["Module"]


class Module(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        cfg = copy.deepcopy(cfg)

        self.optimizer_params = cfg["train"]["optimizer"]
        self.max_chunk_size = cfg["model"].pop("max_chunk_size")
        self.model = MultiResidualBiGRU(**cfg["model"])
        self.eval_time_window = cfg["eval"]["window"]
        self.step_interval = cfg["dataset"]["agg_interval"]

        self.preds_list: list = []
        self.ids_list: list = []
        self.steps_list: list = []
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
        _, ids, steps, y_batch = batch

        pred = self.forward(batch)

        self.preds_list.append(pred.detach().cpu().numpy())
        self.ids_list.append(ids)
        self.steps_list.append(steps.detach().cpu().numpy())
        loss = nn.MSELoss()(pred.float(), y_batch.float())
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
        list_pred = [pred[0] for pred in self.preds_list]
        series_ids = [ids[0] for ids in self.ids_list]
        list_steps = self.steps_list

        event_df = get_event_df("train")
        if len(list_steps) != 0:
            event_df = event_df[event_df["series_id"].isin(np.unique(series_ids))][
                ["series_id", "event", "step", "night"]
            ]
            event_df = event_df.dropna()

        submission_df = pd.concat(
            [
                get_submission_df(
                    pred[np.newaxis],
                    [uid],
                    steps,
                    calc_type="top-probs",
                    step_interval=self.step_interval,
                    time_window=self.eval_time_window,
                )
                for pred, uid, steps in zip(list_pred, series_ids, list_steps)
            ]
        )

        score = calc_event_detection_ap(event_df, submission_df)
        self.log("EventDetectionAP", score, on_epoch=True, logger=True, prog_bar=True)

        self.preds_list.clear()
        self.ids_list.clear()
        self.steps_list.clear()

    def configure_optimizers(self):
        optimizer_params = self.optimizer_params.copy()
        scheduler_params = optimizer_params.pop("scheduler")

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
