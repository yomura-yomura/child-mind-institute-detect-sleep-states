import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the loss function
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, real, output):
        ce = nn.BCEWithLogitsLoss(reduction="none")
        loss = ce(real[:, :, 0:3].unsqueeze(-1), output.unsqueeze(-1))  # Example shape (32, 864, 3)
        mask = real[:, :, 3] * real[:, :, 4]  # Example shape (32, 864)
        mask = mask.to(loss.dtype)
        mask = mask.unsqueeze(-1)  # Example shape (32, 864, 1)
        mask = mask.repeat(1, 1, 3)  # Example shape (32, 864, 3)
        loss *= mask  # Example shape (32, 864, 3)
        return torch.sum(loss) / torch.sum(mask)


# Define the model configuration
CFG = {
    "TPU": 0,
    "block_size": 15552,
    "block_stride": 15552 // 16,
    "patch_size": 18,
    "fog_model_dim": 320,
    "fog_model_num_heads": 6,
    "fog_model_num_encoder_layers": 5,
    "fog_model_num_lstm_layers": 2,
    "fog_model_first_dropout": 0.1,
    "fog_model_encoder_dropout": 0.1,
    "fog_model_mha_dropout": 0.0,
}


# Define the EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=CFG["fog_model_dim"], num_heads=CFG["fog_model_num_heads"], dropout=CFG["fog_model_mha_dropout"]
        )
        self.add = nn.Identity()
        self.layernorm = nn.LayerNorm(CFG["fog_model_dim"])
        self.seq = nn.Sequential(
            nn.Linear(CFG["fog_model_dim"], CFG["fog_model_dim"], activation="relu"),
            nn.Dropout(CFG["fog_model_encoder_dropout"]),
            nn.Linear(CFG["fog_model_dim"], CFG["fog_model_dim"]),
            nn.Dropout(CFG["fog_model_encoder_dropout"]),
        )

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.add(x, attn_output)
        x = self.layernorm(x)
        x = self.add(x, self.seq(x))
        x = self.layernorm(x)
        return x


# Define the FOGEncoder
class FOGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_linear = nn.Linear(CFG["patch_size"] * 3, CFG["fog_model_dim"])
        self.add = nn.Identity()
        self.first_dropout = nn.Dropout(CFG["fog_model_first_dropout"])
        self.enc_layers = nn.ModuleList([EncoderLayer() for _ in range(CFG["fog_model_num_encoder_layers"])])
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(CFG["fog_model_dim"] * 2, return_sequences=True, bidirectional=True)
                for _ in range(CFG["fog_model_num_lstm_layers"])
            ]
        )
        self.sequence_len = CFG["block_size"] // CFG["patch_size"]
        self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_len, CFG["fog_model_dim"]) * 0.02)

    def forward(self, x, training=True):
        x = x / 25.0
        x = self.first_linear(x)
        if training:
            random_pos_encoding = torch.roll(
                self.pos_encoding.repeat(GPU_BATCH_SIZE, 1, 1),
                shifts=torch.randint(-self.sequence_len, 0, (GPU_BATCH_SIZE,)),
                dims=1,
            )
            x = self.add(x, random_pos_encoding)
        else:
            x = self.add(x, self.pos_encoding.repeat(GPU_BATCH_SIZE, 1, 1))
        x = self.first_dropout(x)
        for i in range(CFG["fog_model_num_encoder_layers"]):
            x = self.enc_layers[i](x)
        for i in range(CFG["fog_model_num_lstm_layers"]):
            x, _ = self.lstm_layers[i](x)
        return x


# Define the FOGModel
class FOGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FOGEncoder()
        self.last_linear = nn.Linear(CFG["fog_model_dim"] * 2, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.last_linear(x)
        x = torch.sigmoid(x)
        return x
