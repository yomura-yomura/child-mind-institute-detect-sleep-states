import os.path
import warnings

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

__all__ = ["ModelCheckpointWithSymlinkToBest"]


class ModelCheckpointWithSymlinkToBest(ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best"

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
