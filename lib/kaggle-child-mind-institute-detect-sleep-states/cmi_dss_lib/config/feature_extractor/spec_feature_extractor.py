import dataclasses
from typing import Literal


@dataclasses.dataclass
class SpecFeatureExtractorConfig:
    name: Literal["SpecFeatureExtractor"]
    height: int
    hop_length: int
    win_length: int | None
