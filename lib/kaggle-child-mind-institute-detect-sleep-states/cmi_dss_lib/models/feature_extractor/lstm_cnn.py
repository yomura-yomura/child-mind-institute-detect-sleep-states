
from typing import Callable, Optional, Sequence
from .cnn import CNNSpectrogram
import torch
import torch.nn as nn


class LSTMandCNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        model_dim: int,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_size)
        self.height = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_chans = 1
        self.out_size = out_size
        self.model_dim = model_dim

        if self.out_size is not None:
            if self.model_dim == 1:
                self.pool = nn.AdaptiveAvgPool1d(self.out_size)
            elif self.model_dim == 2:
                self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))
            else:
                raise ValueError(f"unexpected {model_dim=}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor:
                1D: (batch_size, height, time_steps)
                2D: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        if self.out_size is not None:
            # x = x.unsqueeze(1)  # x: (batch_size, 1, in_channels, time_steps)
            x = self.pool(x)  # x: (batch_size, 1, in_channels, time_steps)
            # x = x.squeeze(1)  # x: (batch_size, in_channels, time_steps)

        x = x.transpose(1, 2)  # x: (batch_size, time_steps, in_channels)
        x = self.fc(x)  # x: (batch_size, time_steps, hidden_size)
        x, _ = self.lstm(x)  # x: (batch_size, time_steps, hidden_size * num_directions)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, time_steps)

        if self.model_dim == 2:
            # LSTM Feature Extractorの後にCNNSpectrogramを通しただけ
            x = CNNSpectrogram(in_chanels=self.height)(x)# x: (batch_size, out_chans, hidden_size * num_directions, time_steps)
        return x


