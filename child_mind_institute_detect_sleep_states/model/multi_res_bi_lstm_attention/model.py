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
        duration: int,
        # height: int,
        patch_size: int,
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
        self.duration = duration
        self.patch_size = patch_size
        # self.out_size = mha_embed_dim
        # if self.out_size is not None:
        #     self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        # self.sequence_len = CFG["block_size"] // CFG["patch_size"]
        # self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_len, CFG["fog_model_dim"]) * 0.02)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, in_channels, time_steps)
        x = x.reshape(x.shape[0], x.shape[1], self.duration // self.patch_size, self.patch_size)

        # x = torch.max(x, dim=-1)[0]
        x = torch.mean(x, dim=-1)
        # (batch_size, in_channels, duration // patch_size)

        # x /= 25.0
        x = x.permute(0, 2, 1)  # (batch_size, duration // patch_size, in_channels)
        x = self.first_linear(x)  # (batch_size, duration // patch_size, embed_dim)
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
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, duration // patch_size)
        return x


# Define the FOGModel
class FOGModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        duration: int = 17280,
        patch_size: int = 20,
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
            patch_size=patch_size,
            duration=duration,
            n_features=n_features,
            n_encoder_layers=n_encoder_layers,
            n_lstm_layers=n_lstm_layers,
            dropout=dropout,
            mha_embed_dim=mha_embed_dim,
            mha_n_heads=mha_n_heads,
            mha_dropout=mha_dropout,
        )
        self.last_linear = nn.Linear(duration // patch_size, out_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.last_linear(x)
        # x = torch.sigmoid(x)
        x = x.unsqueeze(1)
        return x
