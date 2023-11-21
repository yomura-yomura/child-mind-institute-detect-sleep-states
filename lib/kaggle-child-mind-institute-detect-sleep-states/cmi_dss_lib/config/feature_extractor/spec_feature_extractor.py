import dataclasses
from typing import Literal

from omegaconf import DictConfig


@dataclasses.dataclass
class SpecFeatureExtractorConfig(DictConfig):
    name: Literal["SpecFeatureExtractor"]
    height: int
    hop_length: int
    win_length: int | None
