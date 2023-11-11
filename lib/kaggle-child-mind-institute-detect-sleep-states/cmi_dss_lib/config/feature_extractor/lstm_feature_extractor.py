import dataclasses
from typing import Literal


@dataclasses.dataclass
class LSTMFeatureExtractorConfig:
    name: Literal["LSTMFeatureExtractor"]
    hidden_size: int
    num_layers: int
    bidirectional: bool
