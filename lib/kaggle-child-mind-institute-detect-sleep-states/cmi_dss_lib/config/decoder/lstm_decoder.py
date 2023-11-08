import dataclasses
from typing import Literal


@dataclasses.dataclass
class LSTMDecoder:
    name: Literal["LSTMDecoder"]
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
