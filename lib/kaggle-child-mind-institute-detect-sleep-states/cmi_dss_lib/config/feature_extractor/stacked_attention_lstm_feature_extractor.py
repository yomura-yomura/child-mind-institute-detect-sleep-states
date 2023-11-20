import dataclasses
from typing import Literal

from omegaconf import DictConfig


@dataclasses.dataclass
class StackedAttentionLSTMFeatureExtractor(DictConfig):
    name: Literal["StackedAttentionLSTMFeatureExtractor"]
    n_encoder_layers: int
    n_lstm_layers: int
    dropout: float
    mha_embed_dim: int
    mha_n_heads: int
    mha_dropout: int
