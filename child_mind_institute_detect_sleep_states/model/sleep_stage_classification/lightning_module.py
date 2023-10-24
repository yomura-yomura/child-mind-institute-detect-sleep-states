import pathlib

import lightning.pytorch as lp
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from ...data.comp_dataset import get_event_df, get_submission_df
from ...score import calc_event_detection_ap
from .dataset import Batch
from .model import CharGRULSTM

project_root_path = pathlib.Path(__file__).parent.parent.parent


__all__ = ["Module"]


class Module(lp.LightningModule):
    def __init__(
        self,
        steps_in_epoch: int,
        *,
        learning_rate: float | None = None,
        onset_weight: float = 1,
        wakeup_weight: float = 1,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        self.model = CharGRULSTM(
            input_size=steps_in_epoch,
            # hidden_size=256,
            # hidden_size2=128,
            hidden_size=128,
            hidden_size2=64,
            output_size=3,
            # seq_len=6
            # ).to(device)
        )
        self.loss_function = torch.nn.BCELoss(weight=torch.tensor([1, onset_weight, wakeup_weight]))

        self.series_id_list = []
        self.pred_list = []

        self.event_df = None

    def forward(self, batch: Batch) -> STEP_OUTPUT:
        records, _, _ = batch
        records = records.to(self.device)

        hidden = self.model.init_hidden(batch_size=records.shape[0]).to(self.device)
        cell = self.model.init_hidden(batch_size=records.shape[0]).to(self.device)

        pred_y, _, _ = self.model(records, hidden, cell)
        return pred_y

    def training_step(self, batch: Batch) -> STEP_OUTPUT:
        _, label, _ = batch
        label = label.to(self.device)
        pred_y = self.forward(batch)
        loss = self.loss_function(pred_y, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch) -> STEP_OUTPUT:
        _, label, series_id = batch
        self.series_id_list.append(series_id)

        label = label.to(self.device)
        pred_y = self.forward(batch)
        self.pred_list.append(pred_y)
        loss = self.loss_function(pred_y, label)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        series_ids = np.concatenate(self.series_id_list)
        self.series_id_list.clear()
        preds = torch.concat(self.pred_list).detach().cpu().numpy()
        self.pred_list.clear()
        print(f"{preds.max(axis=0) = }")

        event_df = get_event_df("train")
        event_df = event_df[event_df["series_id"].isin(np.unique(series_ids))][["series_id", "event", "step", "night"]]
        event_df = event_df.dropna()

        submission_df1 = get_submission_df(preds, series_ids, calc_type="max-along-type")
        submission_df2 = get_submission_df(preds, series_ids, calc_type="top-probs")

        score1 = calc_event_detection_ap(event_df, submission_df1)
        score2 = calc_event_detection_ap(event_df, submission_df2)
        self.log("EventDetectionAP1", score1, prog_bar=True)
        self.log("EventDetectionAP2", score2, prog_bar=True)
        self.log("EventDetectionAP", max(score1, score2), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        assert self.learning_rate is not None
        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        return optimizer
