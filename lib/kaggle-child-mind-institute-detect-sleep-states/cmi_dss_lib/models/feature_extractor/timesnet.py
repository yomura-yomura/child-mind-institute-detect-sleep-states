from typing import Optional

import torch
import torch.nn as nn
from timesnet.model import TimesNet


class TimesNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        height,
        dim_model,
        encoder_layers,
        times_blocks,
        num_kernels,
        dropout,
        dim_fc,
        embed_encoding,
        freq,
        task,
        is_fc,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.height = height
        self.fc = nn.Linear(dim_model, height)
        self.times_net = TimesNet(
            len_input =out_size, # time steps
            enc_in = in_channels, # fature num
            d_model=dim_model, # dimension of model
            embed=embed_encoding,
            freq=freq,
            dropout = dropout,
            e_layers=encoder_layers,
            d_ff = dim_fc,
            top_k = times_blocks,
            num_kernels=num_kernels,
            task = task
                    )
        self.out_chans = 1
        self.is_fc = is_fc
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool1d(self.out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)

        if self.out_size is not None:
            x = self.pool(x)  # x: (batch_size, 1, in_channels, out_size)
        x = self.times_net(x)# x: (batch_size, out_size, d_model)
        # height = d_model

        if self.is_fc:
            # height = self.hegiht
            x = self.fc(x)  # x: (batch_size, out_size, height)
        
        x = x.transpose(1, 2)  # x: (batch_size, height, out_size)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, height, out_size)
        return x

