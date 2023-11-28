import torch
import torch.nn as nn


class WaveBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: int, kernel_size: int):
        super().__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        for dilation_rate in [2**i for i in range(self.num_rates)]:
            self.filter_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate * (kernel_size - 1)) / 2),
                    dilation=dilation_rate,
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class WaveNet(nn.Module):
    def __init__(self, in_channels: int = 3, kernel_size: int = 3, use_last_linear: bool = False):
        super().__init__()

        # self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = WaveBlock(in_channels, 32, 8, kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)

        self.gru = nn.GRU(
            input_size=128, hidden_size=128, num_layers=4, batch_first=True, bidirectional=True
        )

        self.use_last_linear = use_last_linear
        if self.use_last_linear:
            self.fc1 = nn.Linear(256, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, time_steps)

        x = x.permute(0, 2, 1)  # (batch_size, time_steps, in_channels)
        # x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)

        print(x.shape)

        x = x.permute(0, 2, 1)  # (batch_size, in_channels, time_steps)
        x, _ = self.gru(x)

        if self.use_last_linear:
            x = self.fc1(x)

        return x
