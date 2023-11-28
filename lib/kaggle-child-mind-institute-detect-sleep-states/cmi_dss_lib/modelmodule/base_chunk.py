import itertools
import pathlib
from abc import abstractmethod
from typing import Any

import cmi_dss_lib.datamodule.seg
import cmi_dss_lib.models.common
import cmi_dss_lib.utils.metrics
import cmi_dss_lib.utils.post_process
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.optim as optim
import tqdm
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from numpy.typing import NDArray
from omegaconf import DictConfig
from torchvision.transforms.functional import resize
from transformers import get_cosine_schedule_with_warmup

import child_mind_institute_detect_sleep_states.score


class BaseChunkModule(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        val_event_df: pl.DataFrame | None,
        duration: int,
        model_save_dir_path: pathlib.Path | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        if val_event_df is None:
            self.val_event_df = None
        else:
            self.val_event_df = val_event_df.to_pandas()

            if "event_onset" not in self.cfg.labels:
                self.val_event_df = self.val_event_df[self.val_event_df["event"] != "onset"]
            if "event_wakeup" not in self.cfg.labels:
                self.val_event_df = self.val_event_df[self.val_event_df["event"] != "wakeup"]
            assert len(self.val_event_df) > 0
            print(self.val_event_df)

        self.num_time_steps = (
            cmi_dss_lib.datamodule.seg.nearest_valid_size(
                int(duration * cfg.upsample_rate), cfg.downsample_rate
            )
            // cfg.downsample_rate
        )
        self.duration = duration
        self.validation_step_outputs: list = []
        self.predict_step_outputs: list = []
        self.__best_loss = np.inf

        # fit

        self.save_top_k = 2
        self.model_save_dir_path = model_save_dir_path

        self.best_score_paths: list[tuple[float, pathlib.Path]] = []
        self.last_model_path = None

        if self.model_save_dir_path is not None:
            self.best_ckpt_path = self.model_save_dir_path / "best.ckpt"
            self.last_ckpt_path = self.model_save_dir_path / "last.ckpt"
        else:
            self.best_ckpt_path = self.last_ckpt_path = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
                raise FileExistsError(self.best_ckpt_path)
            if self.last_ckpt_path is not None and self.last_ckpt_path.exists():
                raise FileExistsError(self.last_ckpt_path)

    @abstractmethod
    def forward(
        self, batch: dict[str : torch.Tensor], *, do_mixup=False, do_cutmix=False
    ) -> dict[str, torch.Tensor | None]:
        pass

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

    def on_fit_start(self) -> None:
        if self.cfg.val_after_steps > 0:
            print(f"validation will be enabled after {self.cfg.val_after_steps} steps")
            self.trainer.limit_val_batches = 0.0

        if self.cfg.resume_from_checkpoint is not None:
            print(f"[Info] Reset Early Stopping Callback")
            for early_stopping_callback in self.trainer.early_stopping_callbacks:
                early_stopping_callback.best_score = torch.tensor(
                    0,
                    device=early_stopping_callback.best_score.device,
                    dtype=early_stopping_callback.best_score.dtype,
                )
                early_stopping_callback.wait_count = 0

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.global_step > self.cfg.val_after_steps and self.trainer.limit_val_batches == 0:
            self.trainer.limit_val_batches = 1.0
            print(f"enabled validation")

    def _evaluation_step(
        self, batch: dict[str : torch.Tensor], step_outputs: list
    ) -> float | None:
        output = self.forward(batch)
        loss = output["loss"].detach().item() if "loss" in output.keys() else None
        logits = output["logits"]  # (batch_size, n_time_steps, n_classes)

        resized_probs = resize(
            logits.sigmoid().detach().cpu(),
            size=[self.duration, logits.shape[2]],
            antialias=False,
        )

        masks = batch["mask"].detach().cpu()

        series_ids = [key.split("_")[0] for key in batch["key"]]
        resized_probs = resized_probs.numpy()
        assert len(series_ids) == len(resized_probs), (
            len(series_ids),
            len(resized_probs),
        )

        if torch.all(masks):
            step_outputs.append(
                (
                    series_ids,
                    resized_probs,
                )
            )
        else:
            resized_masks = resize(masks.unsqueeze(2), size=[self.duration, 1], antialias=False)
            resized_masks = resized_masks.squeeze(2)

            for (
                series_id,
                resized_mask,
                resized_prob,
            ) in zip(
                series_ids,
                resized_masks,
                resized_probs,
                strict=True,
            ):
                if not torch.all(resized_mask):
                    resized_prob = resized_prob[resized_mask].reshape(-1, resized_probs.shape[-1])
                step_outputs.append(
                    (
                        [series_id],
                        [resized_prob],
                    )
                )
        return loss

    @staticmethod
    def _evaluation_epoch_end(step_outputs: list) -> tuple[str, NDArray[np.float_]]:
        flatten_validation_step_outputs = [
            (
                series_id,
                preds,
            )
            for args_in_batch in step_outputs
            for (
                series_id,
                preds,
            ) in zip(*args_in_batch, strict=True)
        ]
        step_outputs.clear()

        for series_id, g in tqdm.tqdm(
            itertools.groupby(
                flatten_validation_step_outputs,
                key=lambda output: output[0],
            ),
            desc="calc val sub_df",
            total=np.unique([series_id for series_id, *_ in flatten_validation_step_outputs]).size,
        ):
            yield (
                series_id,
                np.concatenate([preds.reshape(-1, preds.shape[-1]) for _, preds in g], axis=0),
            )

    def validation_step(self, batch: dict[str : torch.Tensor]):
        loss = self._evaluation_step(batch, self.validation_step_outputs)
        self.log(
            f"valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        n_jobs = -1
        # n_jobs = 1
        # sub_df_list = []
        sub_df = pd.concat(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(cmi_dss_lib.utils.post_process.post_process_for_seg)(
                    keys=[series_id] * len(preds),
                    preds=preds,
                    labels=list(self.cfg.labels),
                    downsample_rate=self.cfg.downsample_rate,
                    score_th=self.cfg.post_process.score_th,
                    distance=self.cfg.post_process.distance,
                )
                for series_id, preds in self._evaluation_epoch_end(self.validation_step_outputs)
            )
        )
        # for series_id, preds in self._evaluation_epoch_end(self.validation_step_outputs):
        #     sub_df_list.append(
        #         cmi_dss_lib.utils.post_process.post_process_for_seg(
        #             keys=[series_id] * len(preds),
        #             preds=preds,
        #             labels=list(self.cfg.labels),
        #             downsample_rate=self.cfg.downsample_rate,
        #             score_th=self.cfg.post_process.score_th,
        #             distance=self.cfg.post_process.distance,
        #         )
        #     )
        # sub_df = pd.concat(sub_df_list)

        # score = cmi_dss_lib.utils.metrics.event_detection_ap(self.val_event_df, sub_df)
        score = child_mind_institute_detect_sleep_states.score.calc_event_detection_ap(
            self.val_event_df,
            sub_df,
            n_jobs=n_jobs
            # calc_type="normal"
        )

        self.log(
            "EventDetectionAP", score, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        if self.model_save_dir_path is not None:
            self.save_checkpoint_top_k(score)

    def save_checkpoint_top_k(self, score: float):
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step

        current_model_path = self.model_save_dir_path / (
            "-".join(
                [
                    f"{epoch=}",
                    f"{step=}",
                    f"EventDetectionAP={score:.4f}",
                ]
            )
            + ".ckpt"
        )

        self.last_model_path = current_model_path

        self.best_score_paths.append((score, current_model_path))
        best_score_paths_in_descending_order = sorted(
            self.best_score_paths, key=lambda pair: pair[0]
        )[::-1]

        monitor = "EventDetectionAP"

        best_model_score, best_model_path = best_score_paths_in_descending_order[0]

        self.trainer.save_checkpoint(current_model_path)

        if len(best_score_paths_in_descending_order) < self.save_top_k or current_model_path in (
            path for _, path in best_score_paths_in_descending_order[: self.save_top_k]
        ):
            print(
                f"Epoch {epoch:d}, global step {step:d}: {monitor!r} reached {score:0.5f}"
                f" (best {best_model_score:0.5f}), saving model to {current_model_path} in top {self.save_top_k}"
            )
            self.best_ckpt_path.unlink(missing_ok=True)
            self.best_ckpt_path.symlink_to(best_model_path)
        else:
            print(
                f"Epoch {epoch:d}, global step {step:d}: {monitor!r} was not in top {self.save_top_k}"
            )

        if len(best_score_paths_in_descending_order) > 0:
            self.last_ckpt_path.unlink(missing_ok=True)
            self.last_ckpt_path.symlink_to(self.last_model_path)

        indices_to_remove = []
        for i, (_, model_path) in enumerate(
            best_score_paths_in_descending_order[self.save_top_k :]
        ):
            if model_path == self.last_model_path:
                continue
            model_path.unlink(missing_ok=True)
            indices_to_remove.append(self.save_top_k + i)

        for i in indices_to_remove[::-1]:
            best_score_paths_in_descending_order.pop(i)
        self.best_score_paths = best_score_paths_in_descending_order

    def predict_step(
        self, batch: dict[str : torch.Tensor]
    ) -> list[tuple[str, NDArray[np.float_]]]:
        step_outputs = []
        self._evaluation_step(batch, step_outputs)
        return step_outputs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
