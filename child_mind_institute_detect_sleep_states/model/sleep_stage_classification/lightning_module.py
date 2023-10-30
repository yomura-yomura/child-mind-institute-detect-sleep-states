import pathlib

import lightning.pytorch as lp
import numpy as np
import pandas as pd
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
        n_features: int,
        *,
        learning_rate: float | None = None,
        onset_weight: float = 1,
        wakeup_weight: float = 1,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        self.model = CharGRULSTM(
            input_size=n_features,
            # hidden_size=256,
            # hidden_size2=128,
            hidden_size=128,
            hidden_size2=64,
            output_size=2,
            # seq_len=6
            # ).to(device)
        )
        self.loss_function = torch.nn.BCELoss(weight=torch.tensor([onset_weight, wakeup_weight]))

        self.series_id_list = []
        self.step_list = []
        self.pred_list = []

        self.event_df = None

    def forward(self, batch: Batch) -> STEP_OUTPUT:
        records, _, _, _ = batch
        # records = records.to(self.device)

        hidden = self.model.init_hidden(batch_size=records.shape[0]).to(self.device)
        cell = self.model.init_hidden(batch_size=records.shape[0]).to(self.device)

        # max_chunk_size = 700_000
        max_chunk_size = 500_000
        n_series = records.shape[1]

        pred_y_list = []

        edges = np.arange(np.ceil(n_series / max_chunk_size) + 1, dtype=np.uint16) * max_chunk_size
        for l_edge, r_edge in zip(edges[:-1], edges[1:]):
            pred_y, hidden, cell = self.model(records[:, l_edge:r_edge, :], hidden, cell)
            pred_y_list.append(pred_y)
        pred_y = torch.concat(pred_y_list, dim=1)

        return pred_y

    def training_step(self, batch: Batch) -> STEP_OUTPUT:
        _, label, _, _ = batch
        label = label.to(self.device)
        pred_y = self.forward(batch)
        loss = self.loss_function(pred_y, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch) -> STEP_OUTPUT:
        _, labels, series_id, steps = batch
        self.series_id_list.append(series_id)
        self.step_list.append(steps.detach().cpu())

        # labels = labels.to(self.device)
        pred_y = self.forward(batch)
        self.pred_list.append(pred_y.detach().cpu())
        loss = self.loss_function(pred_y, labels)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        submission1_df_list = []
        submission2_df_list = []
        for series_ids, steps, preds in zip(self.series_id_list, self.step_list, self.pred_list):
            steps = steps.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            print(f"{preds.max(axis=1) = }")

            submission1_df_list.append(get_submission_df(preds, series_ids, steps, calc_type="max-along-type"))
            submission2_df_list.append(get_submission_df(preds, series_ids, steps, calc_type="top-probs"))

        submission_df1 = pd.concat(submission1_df_list)
        submission_df2 = pd.concat(submission2_df_list)

        event_df = get_event_df("train")
        event_df = event_df[event_df["series_id"].isin(np.unique(self.series_id_list))][
            ["series_id", "event", "step", "night"]
        ]
        event_df = event_df.dropna()

        self.series_id_list.clear()
        self.step_list.clear()
        self.pred_list.clear()

        score1 = calc_event_detection_ap(event_df, submission_df1)
        score2 = calc_event_detection_ap(event_df, submission_df2)
        self.log("EventDetectionAP1", score1, prog_bar=True)
        self.log("EventDetectionAP2", score2, prog_bar=True)
        self.log("EventDetectionAP", max(score1, score2), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        assert self.learning_rate is not None
        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        scheduler_params = {"T_max": 4440, "eta_min": 2e-8}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)

        return [optimizer], [scheduler]
