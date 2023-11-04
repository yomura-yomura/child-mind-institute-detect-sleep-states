import torch
import torch.nn as nn


# Define the EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.seq = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        x = self.layer_norm(x)
        x = x + self.seq(x)
        x = self.layer_norm(x)
        return x


"""
FOGEncoder is a combination of transformer encoder (D=320, H=6, L=5) and two BidirectionalLSTM layers
"""

from ..multi_res_bi_lstm.model import ResidualBiLSTM


class FOGEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_encoder_layers: int,
        n_lstm_layers: int,
        dropout: float,
        mha_embed_dim: int,
        mha_n_heads: int,
        mha_dropout: float,
    ):
        super().__init__()
        self.first_linear = nn.Linear(n_features, mha_embed_dim)
        self.first_dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim=mha_embed_dim, n_heads=mha_n_heads, dropout=mha_dropout)
                for _ in range(n_encoder_layers)
            ]
        )
        self.lstm_layers = nn.ModuleList(
            [
                # nn.LSTM(2 * mha_embed_dim, hidden_size=2 * mha_embed_dim, batch_first=True, bidirectional=True)
                ResidualBiLSTM(mha_embed_dim)
                for _ in range(n_lstm_layers)
            ]
        )
        # self.sequence_len = CFG["block_size"] // CFG["patch_size"]
        # self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_len, CFG["fog_model_dim"]) * 0.02)

    def forward(self, x):
        # x /= 25.0
        x = self.first_linear(x)
        # if training:  # augmentation by randomly roll of the position encoding tensor
        #     random_pos_encoding = torch.roll(
        #         self.pos_encoding.repeat(GPU_BATCH_SIZE, 1, 1),
        #         shifts=torch.randint(-self.sequence_len, 0, (GPU_BATCH_SIZE,)),
        #         dims=1,
        #     )
        #     x = x + random_pos_encoding
        # else:
        # x = x + self.pos_encoding.repeat(GPU_BATCH_SIZE, 1, 1)

        x = self.first_dropout(x)
        for layer in self.enc_layers:
            x = layer(x)
        for layer in self.lstm_layers:
            x, _ = layer(x)
        return x


# Define the FOGModel
class FOGModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_encoder_layers: int = 5,
        n_lstm_layers: int = 2,
        out_size: int = 2,
        dropout: float = 0.1,
        mha_embed_dim: int = 320,
        mha_n_heads: int = 5,
        mha_dropout: float = 0,
    ):
        super().__init__()
        self.encoder = FOGEncoder(
            n_features=n_features,
            n_encoder_layers=n_encoder_layers,
            n_lstm_layers=n_lstm_layers,
            dropout=dropout,
            mha_embed_dim=mha_embed_dim,
            mha_n_heads=mha_n_heads,
            mha_dropout=mha_dropout,
        )
        self.last_linear = nn.Linear(mha_embed_dim, out_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.last_linear(x)
        # x = torch.sigmoid(x)
        return x
