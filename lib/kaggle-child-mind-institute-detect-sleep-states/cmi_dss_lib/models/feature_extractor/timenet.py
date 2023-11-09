from typing import Optional
from times-net.model import TimesNet
import torch
import torch.nn as nn


class TimeNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        num_layers,
        bidirectional,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_size)
        self.height = hidden_size * (2 if bidirectional else 1)
        self.time_net = TimesNet(
            input_size = hidden_size,
            hidden_size=hidden_size,
            out_size=out_size,
            n_layers=num_layers,
            bidir=bidirectional,
                    )
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        if self.out_size is not None:
            x = x.unsqueeze(1)  # x: (batch_size, 1, in_channels, time_steps)
            x = self.pool(x)  # x: (batch_size, 1, in_channels, output_size)
            x = x.squeeze(1)  # x: (batch_size, in_channels, output_size)
        x = x.transpose(1, 2)  # x: (batch_size, output_size, in_channels)
        x = self.fc(x)  # x: (batch_size, output_size, hidden_size)
        x, _ = self.stacked_gru(x)  # x: (batch_size, output_size, hidden_size * num_directions)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, output_size)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, hidden_size * num_directions, time_steps)
        return x


#####
# child_mind_institute_detect_sleep_states/model/multi_res_bi_gru/model.py
#####

class ResidualBiLSTM(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.time_net = TimesNet(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(hidden_size * dir_factor, hidden_size * dir_factor * 2)
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None, c=None):
        if h is None and c is None:
            H = None
        else:
            H = (h, c)
        res, (new_h, new_c) = self.lstm(x, H)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, (new_h, new_c)


class MultiResidualBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers, bidir=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.bidir = bidir

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bilstm = nn.ModuleList([ResidualBiLSTM(hidden_size, n_layers=1, bidir=bidir) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_size, hidden_size*2)

    def forward(self, x, h=None, c=None):
        # if we are at the beginning of a sequence (no hidden state)
        num_directions = 2 if self.bidir else 1
        if h is None:
            # (re)initialize the hidden state
            h = [torch.zeros(num_directions, x.shape[0], self.hidden_size, device="cuda") for _ in range(self.n_layers)]

        if c is None:
            # (re)initialize the hidden state
            c = [torch.zeros(num_directions, x.shape[0], self.hidden_size, device="cuda") for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        new_c = []
        for i, res_bilstm in enumerate(self.res_bilstm):
            x, (new_hi, new_ci) = res_bilstm(x, h[i], c[i])
            new_h.append(new_hi)
            new_c.append(new_ci)

        x = self.fc_out(x)
        #         x = F.normalize(x,dim=0)
        return x, (new_h, new_c)  # log probabilities + hidden states
