from typing import Literal
import dataclasses


@dataclasses.dataclass
class TransformerDecoder:
    name: Literal["TransformerDecoder"]
    hidden_size: int
    num_layers: int
    nhead: int
    dropout: float
