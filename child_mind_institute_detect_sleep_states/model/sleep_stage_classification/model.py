import pathlib

import torch
import torch.nn as nn
from torch.autograd import Variable

project_root_path = pathlib.Path(__file__).parent.parent.parent.parent

__all__ = ["CharGRULSTM"]


class CharGRULSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_size2,
        output_size,
        rnn_type="lstm",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.rnntype = rnn_type

        self.embeddingLayer = nn.Linear(self.input_size, self.hidden_size)
        # self.embeddingLayer = nn.Conv1d(
        #     self.input_size, self.hidden_size, kernel_size=1
        # )

        if rnn_type == "gru":
            self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size, bias=True)
        else:
            self.rnn = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                batch_first=True,
                bidirectional=True,
            )
            # self.rnn = nn.LSTMCell(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=0.5)

        self.hidden2 = nn.Linear(
            # self.hidden_size,
            self.hidden_size * 2,
            self.hidden_size2,
        )
        # self.hidden2 = nn.Conv1d(self.hidden_size, self.hidden_size2, kernel_size=8)

        # Only take the output from the final timestep
        self.outputLayer = nn.Linear(self.hidden_size2, self.output_size)

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # self.outputLayer = nn.Linear(self.hidden_size*self.seq_len, self.output_size)

        # self.softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)

        self.predloss = nn.CrossEntropyLoss()

    def forward(self, batch, hidden, cell=None):
        batch_emb = self.embeddingLayer(batch)
        batch_emb = self.dropout(batch_emb)
        if self.rnntype == "lstm":
            # output,(hidden, cell) = self.rnn(batchemb.view(batchemb.shape[0],-1,self.hidden_size), (hidden, cell))
            output, (hidden, cell) = self.rnn(batch_emb, (hidden, cell))
            # hidden, cell = self.rnn(batchemb, (hidden, cell))
        else:
            hidden = self.rnn(batch_emb, hidden)
        # Only take the output from the final timestep
        # output = self.outputLayer(self.dropout(output[:,-1,:]))
        # output = self.hidden2(output[:, -1, :])
        output = self.hidden2(output)
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # output = self.outputLayer(self.dropout(output.view(-1,self.seq_len*self.hidden_size)))
        output = self.outputLayer(output)
        # output = self.softmax(output)
        if self.rnntype == "lstm":
            return output, hidden, cell
        else:
            return output, hidden

    def init_hidden(self, batch_size):
        # return Variable(torch.randn(1, batch_size, self.hidden_size))
        return Variable(torch.randn(2, batch_size, self.hidden_size))
        # return torch.randn(2, batch_size, self.hidden_size)
