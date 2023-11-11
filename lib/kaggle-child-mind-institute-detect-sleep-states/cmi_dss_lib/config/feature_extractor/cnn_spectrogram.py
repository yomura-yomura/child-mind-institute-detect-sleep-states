import dataclasses
from typing import Literal


@dataclasses.dataclass
class CNNSpectrogramConfig:
    name: Literal["CNNSpectrogram"]
    base_filters: int
    kernel_sizes: list[int]
    stride: int
    sigmoid: bool
    reinit: bool
