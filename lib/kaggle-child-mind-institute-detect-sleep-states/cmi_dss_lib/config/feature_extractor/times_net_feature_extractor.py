import dataclasses
from typing import Literal

from omegaconf import DictConfig


@dataclasses.dataclass
class TimesNetFeatureExtractor(DictConfig):
    name: Literal["TimesNetFeatureExtractor"]
    height: int
    dim_model: int
    encoder_layers: int
    times_blocks: int
    num_kernels: int
    dropout: float
    dim_fc: int
    task: Literal["classification", "forecast", "anomaly"]

    is_fc: bool
    embed_encoding: Literal["timeF"]
    freq: Literal["s"]
