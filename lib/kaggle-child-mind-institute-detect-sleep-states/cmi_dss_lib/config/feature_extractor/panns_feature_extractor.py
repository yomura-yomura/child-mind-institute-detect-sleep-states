import dataclasses
from typing import Literal

from omegaconf import DictConfig


@dataclasses.dataclass
class PANNsFeatureExtractorConfig(DictConfig):
    name: Literal["PANNsFeatureExtractor"]
    base_filters: int
    kernel_sizes: list[int]
    stride: int
    sigmoid: bool
    reinit: bool
    win_length: int | None
