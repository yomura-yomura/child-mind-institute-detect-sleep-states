from typing import Literal
import dataclasses


@dataclasses.dataclass
class LSTMFeatureExtractorConfig:
    name: Literal["LSTMFeatureExtractor"]
    hidden_size: int
    num_layers: int
    bidirectional: bool
