import os.path
import warnings

import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import *
from lightning.pytorch.callbacks.model_checkpoint import _PATH
from torch import Tensor

__all__ = ["ModelCheckpointWithSymlinkToBest"]


class ModelCheckpointWithSymlinkToBest(ModelCheckpoint):
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

    def _save_last_checkpoint(self, trainer: L.Trainer, monitor_candidates: dict[str, Tensor]) -> None:
        """
        save last+best checkpoint
        """
        if trainer.global_step <= self.val_after_steps:
            return

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
