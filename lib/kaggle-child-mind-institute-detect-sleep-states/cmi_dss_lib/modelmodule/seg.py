from typing import Any, Optional

import cmi_dss_lib.datamodule.seg
import cmi_dss_lib.models.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.optim as optim
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
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

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.global_step > self.cfg.val_after_steps and self.trainer.limit_val_batches == 0:
            self.trainer.limit_val_batches = 1.0
            print(f"enabled validation")

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
        masks = batch["mask"].detach().cpu()[:, ::n_interval].unsqueeze(2)

        series_ids = [key.split("_")[0] for key in batch["key"]]
        resized_labels = resized_labels.numpy()
        resized_probs = resized_probs.numpy()
        assert len(series_ids) == len(resized_labels) == len(resized_probs), (
            len(series_ids),
            len(resized_labels),
            len(resized_probs),
        )

        if torch.all(masks):
            self.validation_step_outputs.append((series_ids, resized_labels, resized_probs))
        else:
            resized_masks = resize(masks, size=[self.duration, 1], antialias=False)
            resized_masks = resized_masks.squeeze(2)

            for series_id, resized_mask, resized_label, resized_prob in zip(
                series_ids, resized_masks, resized_labels, resized_probs, strict=True
            ):
                if not torch.all(resized_mask):
                    resized_label = resized_label[resized_mask].reshape(-1, 3)
                    resized_prob = resized_prob[resized_mask].reshape(-1, 3)
                self.validation_step_outputs.append(([series_id], [resized_label], [resized_prob]))

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
        if len(self.validation_step_outputs) == 0:
            return

        flatten_validation_step_outputs = [
            (series_id, labels, preds)
            for args_in_batch in self.validation_step_outputs
            for series_id, labels, preds in zip(*args_in_batch, strict=True)
        ]
        self.validation_step_outputs.clear()

        import itertools

        import tqdm

        sub_df_list = []
        for series_id, g in tqdm.tqdm(
            itertools.groupby(
                flatten_validation_step_outputs,
                key=lambda output: output[0],
            ),
            desc="calc val sub_df",
            total=np.unique([series_id for series_id, *_ in flatten_validation_step_outputs]).size,
        ):
            preds = np.concatenate([preds.reshape(-1, 3) for _, _, preds in g], axis=0)

            sub_df_list.append(
                cmi_dss_lib.utils.post_process.post_process_for_seg(
                    keys=[series_id] * len(preds),
                    preds=preds,
                    downsample_rate=self.cfg.downsample_rate,
                    score_th=self.cfg.post_process.score_th,
                    distance=self.cfg.post_process.distance,
                )
            )
        sub_df = pd.concat(sub_df_list)

        score = cmi_dss_lib.utils.metrics.event_detection_ap(self.val_event_df.to_pandas(), sub_df)
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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
