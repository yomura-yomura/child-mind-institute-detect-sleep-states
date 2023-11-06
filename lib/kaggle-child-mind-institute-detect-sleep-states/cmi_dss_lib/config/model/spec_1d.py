import dataclasses
from typing import Literal


@dataclasses.dataclass
class Spec1DConfig:
    name: Literal["Spec1D"]
