import dataclasses
from typing import Literal

from omegaconf import DictConfig

from .dir import DirConfig

__all__ = ["PrepareDataConfig"]


@dataclasses.dataclass
class PrepareDataConfig(DictConfig):
    dir: DirConfig

    phase: Literal["train", "test"]

    scale_type: Literal["constant", "robust_scaler"]

    save_as_npz: bool

    just_load_scaler: bool

    rolling_features: Literal[
        "n_unique_anglez_5min",
        "n_unique_enmo_5min",
        "rolling_unique_anglez_sum",
        "rolling_unique_enmo_sum",
    ]
