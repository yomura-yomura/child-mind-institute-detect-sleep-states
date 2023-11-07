from typing import Literal
import dataclasses


@dataclasses.dataclass
class SpecFeatureExtractorConfig:
    name: Literal["SpecFeatureExtractor"]
    height: int
    hop_length: int
    win_length: int | None
