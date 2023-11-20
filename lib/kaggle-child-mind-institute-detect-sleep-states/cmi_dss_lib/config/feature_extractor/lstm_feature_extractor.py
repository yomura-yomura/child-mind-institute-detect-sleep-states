import dataclasses
from typing import Literal

from omegaconf import DictConfig


@dataclasses.dataclass
class LSTMFeatureExtractorConfig(DictConfig):
    name: Literal["LSTMFeatureExtractor"]
    hidden_size: int
    num_layers: int
    bidirectional: bool
