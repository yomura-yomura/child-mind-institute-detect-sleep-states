from typing import Optional

import cmi_dss_lib.datamodule.seg
import cmi_dss_lib.models.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import polars as pl
import torch
import torch.optim as optim
from lightning import LightningModule
from torchvision.transforms.functional import resize
from transformers import get_cosine_schedule_with_warmup

from ..config import TrainConfig


class SegModel(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        self.num_time_steps = (
            cmi_dss_lib.datamodule.seg.nearest_valid_size(
                int(duration * cfg.upsample_rate), cfg.downsample_rate
            )
            // cfg.downsample_rate
        )
        self.model = cmi_dss_lib.models.common.get_model(
            cfg, feature_dim=feature_dim, n_classes=num_classes, num_time_steps=self.num_time_steps
        )
        self.duration = duration
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf

    def forward(
        self, batch: dict[str : torch.Tensor], *, do_mixup=False, do_cutmix=False
    ) -> dict[str, Optional[torch.Tensor]]:
        return self.model(batch["feature"], batch["label"], do_mixup, do_cutmix)

    def training_step(self, batch):
        do_mixup = np.random.rand() < self.cfg.augmentation.mixup_prob
        do_cutmix = np.random.rand() < self.cfg.augmentation.cutmix_prob

        output = self.forward(batch, do_mixup=do_mixup, do_cutmix=do_cutmix)
        loss: torch.Tensor = output["loss"]
        # logits = output["logits"]  # (batch_size, n_time_steps, n_classes)
        self.log(
            f"train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: dict[str : torch.Tensor]):
        output = self.forward(batch)
        loss = output["loss"]
        logits = output["logits"]  # (batch_size, n_time_steps, n_classes)

        resized_probs = resize(
            logits.sigmoid().detach().cpu(),
            size=[self.duration, logits.shape[2]],
            antialias=False,
        )
        resized_labels = resize(
            batch["label"].detach().cpu(),
            size=[self.duration, logits.shape[2]],
            antialias=False,
        )

        n_interval = int(1 / (self.num_time_steps / self.cfg.duration))
        mask = batch["mask"].detach().cpu()[::n_interval]
        if not np.all(mask):
            resized_masks = resize(mask, size=[self.duration, logits.shape[2]], antialias=False)
            resized_probs = resized_probs[resized_masks]
            resized_labels = resized_labels[resized_masks]

        self.validation_step_outputs.append(
            (
                batch["key"],
                resized_labels.numpy(),
                resized_probs.numpy(),
                loss.detach().item(),
            )
        )
        self.log(
            f"valid_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        # labels = np.concatenate([x[1] for x in self.validation_step_outputs])
        preds = np.concatenate([x[2] for x in self.validation_step_outputs])
        # losses = np.array([x[3] for x in self.validation_step_outputs])
        # loss = losses.mean()

        val_pred_df = cmi_dss_lib.utils.post_process.post_process_for_seg(
            keys=keys,
            preds=preds,
            downsample_rate=self.cfg.downsample_rate,
            score_th=self.cfg.post_process.score_th,
            distance=self.cfg.post_process.distance,
        )
        score = cmi_dss_lib.utils.metrics.event_detection_ap(
            self.val_event_df.to_pandas(), val_pred_df.to_pandas()
        )
        self.log(
            "EventDetectionAP", score, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )

        # if loss < self.__best_loss:
        #     np.save("keys.npy", np.array(keys))
        #     np.save("labels.npy", labels)
        #     np.save("preds.npy", preds)
        #     val_pred_df.write_csv("val_pred_df.csv")
        #     torch.save(self.model.state_dict(), "best_model.pth")
        #     print(f"Saved best model {self.__best_loss} -> {loss}")
        #     self.__best_loss = loss

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
