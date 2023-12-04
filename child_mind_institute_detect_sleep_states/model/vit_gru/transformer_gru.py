import datetime
import inspect

import torch
import torch.nn as nn

torch.manual_seed(datetime.datetime.now().microsecond)


# send only the relevant params to the model
def getParams(cls, params):
    p = {k: params[k] for k in set(inspect.getfullargspec(cls).args) & set(params.keys())}
    print(p)
    return p


# sinusioid positional encoding, for given sequence length and embedding size, and tunable parameter n
def getPositionalEncoding(seq_len, d_model, n=10000):
    pos = torch.arange(0, seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2).unsqueeze(0)
    enc = torch.zeros(seq_len, d_model)
    enc[:, 0::2] = torch.sin(pos / n ** (i / d_model))
    enc[:, 1::2] = torch.cos(pos / n ** (i / d_model))
    return enc


# alibi bias, see paper; may not help;
def alibi_bias(b, a=1):
    b = torch.zeros_like(b)
    n = b.shape[0] // 2 + 1
    for h in range(0, 8):
        bias = -1 / 2 ** (h + a + 1) * torch.arange(0, n)
        b[:n, h] = torch.flip(bias, [0])
        b[n - 1 :, h] = bias
    return b


# a patch that isn't a patch :(
class IdentityPatch(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        return x


# flipped channel ;
class GroupNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(*args, **kwargs)

    def forward(self, x):
        assert len(x.shape) in [2, 3]
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x
        x = self.norm(x)
        x = x.permute(0, 2, 1) if len(x.shape) == 3 else x
        return x
