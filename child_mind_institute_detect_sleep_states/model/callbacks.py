import os.path
import warnings

import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import *
from lightning.pytorch.callbacks.model_checkpoint import _PATH
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint as _ModelCheckpoint
from torch import Tensor

__all__ = ["ModelCheckpoint"]


class ModelCheckpoint(_ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best"

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        val_after_steps: int = 0,
    ):
        self.val_after_steps = val_after_steps
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

    def _should_skip_saving_checkpoint(self, trainer: L.Trainer) -> bool:
        return super()._should_skip_saving_checkpoint(trainer) or trainer.global_step <= self.val_after_steps

    def _save_last_checkpoint(self, trainer: L.Trainer, monitor_candidates: dict[str, Tensor]) -> None:
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
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_BEST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.previous_best_model_path = getattr(self, "previous_best_model_path", ""), filepath
        if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
            if not os.path.join(self.best_model_path):
                warnings.warn(f"'{self.best_model_path}' not found")
            else:
                self._link_checkpoint(trainer, self.best_model_path, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)


from lightning.pytorch.callbacks.early_stopping import EarlyStopping as _EarlyStopping
from lightning.pytorch.callbacks.early_stopping import *


class EarlyStopping(_EarlyStopping):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
        val_after_steps: int = 0,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
        self.val_after_steps = val_after_steps

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        return super()._should_skip_check(trainer) | trainer.global_step <= self.val_after_steps
