import dataclasses
from typing import Literal


@dataclasses.dataclass
class PANNsFeatureExtractorConfig:
    name: Literal["PANNsFeatureExtractor"]
    base_filters: int
    kernel_sizes: list[int]
    stride: int
    sigmoid: bool
    reinit: bool
    win_length: int | None
