import dataclasses
from typing import Literal

from omegaconf import DictConfig


@dataclasses.dataclass
class CNNSpectrogramConfig(DictConfig):
    name: Literal["CNNSpectrogram"]
    base_filters: int
    kernel_sizes: list[int]
    stride: int
    sigmoid: bool
    reinit: bool
