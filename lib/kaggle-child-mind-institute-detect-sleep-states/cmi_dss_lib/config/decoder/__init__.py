from typing import TypeAlias

from .lstm_decoder import LSTMDecoder
from .mlp_decoder import MLPDecoder
from .transformer_decoder import TransformerDecoder
from .unet_1d_decoder import UNet1DDecoder

Decoder: TypeAlias = LSTMDecoder | MLPDecoder | TransformerDecoder | UNet1DDecoder
