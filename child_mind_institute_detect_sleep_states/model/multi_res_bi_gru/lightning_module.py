# import pathlib
# import sys

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

# project_root_path = pathlib.Path(__file__).parent.parent.parent.parent
# sys.path.append(str(project_root_path / "lib" / "jumtras"))
# from lib.jumtras.child_sleep.model.multi_res_bi_gru import ModelModule as Module


class Module(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        cfg = copy.deepcopy(cfg)

        self.optimizer_params = cfg["train"]["optimizer"]
        self.max_chunk_size = cfg["model"].pop("max_chunk_size")
        self.model = MultiResidualBiGRU(**cfg["model"])
        self.list_pred: list = []
        self.list_ids: list = []
        self.list_steps: list = []
        self.list_pred_train: list = []
        self.list_ids_train: list = []
        self.list_len_train: list = []
        self.save_hyperparameters()

    def forward(self, batch):
        X_batch, y_batch, ids, _ = batch

        y_batch = y_batch.to(self.device, non_blocking=True)
        pred = torch.zeros(y_batch.shape).to(self.device, non_blocking=True)

        h = None
        seq_len = X_batch.shape[1]

        for i in range(0, seq_len, self.max_chunk_size):
            X_chunk = X_batch[:, i : i + self.max_chunk_size].float().to(self.device, non_blocking=True)
            y_pred, h = self.model(X_chunk, h)
            h = [hi.detach() for hi in h]
            pred[:, i : i + self.max_chunk_size] = y_pred
        return pred

    def training_step(self, batch, batch_idx):
        _, y_batch, _, _ = batch
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
        _, y_batch, ids, steps = batch

        pred = self.forward(batch)

        self.list_pred.append(pred.float())
        self.list_ids.append(ids)
        self.list_steps.append(steps.detach().cpu().numpy())
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

    def predict(self, X_batch):
        pred = torch.zeros(torch.Size([1, X_batch.shape[1], 2])).to(self.device, non_blocking=True)

        h = None
        seq_len = X_batch.shape[1]

        for i in range(0, seq_len, self.max_chunk_size):
            X_chunk = X_batch[:, i : i + self.max_chunk_size].float().to(self.device, non_blocking=True)
            y_pred, h = self.model(X_chunk, h)
            h = [hi.detach() for hi in h]
            pred[:, i : i + self.max_chunk_size] = y_pred
        return pred

    def on_validation_epoch_end(self) -> None:
        list_pred = [pred.detach().cpu().numpy()[0] for pred in self.list_pred]
        series_ids = [l[0] for l in self.list_ids]
        list_steps = self.list_steps

        event_df = get_event_df("train")
        if len(list_steps) != 0:
            event_df = event_df[event_df["series_id"].isin(np.unique(series_ids))][
                ["series_id", "event", "step", "night"]
            ]
            event_df = event_df.dropna()

        # submission_df1 = get_submission_df(list_pred, series_ids, list_steps, calc_type="max-along-type")
        submission_df2 = pd.concat(
            [
                get_submission_df(pred[np.newaxis], [uid], steps, calc_type="top-probs")
                for pred, uid, steps in zip(list_pred, series_ids, list_steps)
            ]
        )

        # score1 = calc_event_detection_ap(event_df, submission_df1)
        score2 = calc_event_detection_ap(event_df, submission_df2)
        # self.log("EventDetectionAP1", score1, on_epoch=True, logger=True, prog_bar=True)
        # self.log("EventDetectionAP2", score2, on_epoch=True, logger=True, prog_bar=True)
        # self.log("EventDetectionAP", max(score1, score2), on_epoch=True, logger=True, prog_bar=True)
        self.log("EventDetectionAP", score2, on_epoch=True, logger=True, prog_bar=True)

        self.list_pred.clear()
        self.list_ids.clear()
        self.list_steps.clear()

    def configure_optimizers(self):
        optimizer_params = self.optimizer_params.copy()
        scheduler_params = optimizer_params.pop("scheduler")

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
