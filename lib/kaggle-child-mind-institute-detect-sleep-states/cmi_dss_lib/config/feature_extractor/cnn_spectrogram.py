from typing import Literal
import dataclasses


@dataclasses.dataclass
class CNNSpectrogramConfig:
    name: Literal["CNNSpectrogram"]
    base_filters: int
    kernel_sizes: list[int]
    stride: int
    sigmoid: bool
    reinit: bool
