import dataclasses
from typing import Literal


@dataclasses.dataclass
class UNet1DDecoder:
    name: Literal["UNet1DDecoder"]
    bilinear: bool
    se: bool
    res: bool
    scale_factor: int
    dropout: float
