import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from .layers.Conv_Blocks import Inception_Block_V1
from .layers.Embed import DataEmbedding


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len,top_k,d_model,d_ff,num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = 0
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, len_input:int,enc_in:int,d_model:int,embed:str,freq:str,dropout:float,e_layers:int,top_k,d_ff:int,num_kernels:int,task:str):
        """Parameter for TimesNet

        Parameters
        ----------
        
        time_steps : int
            input sequence length
        enc_in : int
            encoder input size
        d_model : int
            dimension of model
        embed : str
            time features encoding, options:[timeF, fixed, learned]
        freq : str
            freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        dropout : float
            dropout
        e_layers : int
            num of encoder layers
        top_k : _type_
            for TimesBlock
        d_ff : int
            dimension of fcn
        num_kernels : int
            for Inception
        """
        super(TimesNet, self).__init__()
        self.task_name = task #"anomaly_detection" #'imputation' 'anomaly_detection':
        self.seq_len = len_input # input sequence length
        self.pred_len = 0 # predict sequence length
        self.model = nn.ModuleList([TimesBlock(len_input,top_k,d_model,d_ff,num_kernels)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model,len_input, embed, freq,
                                           dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        ## porject back
        #dec_out = self.projection(enc_out)

        ## De-Normalization from Non-stationary Transformer
        #dec_out = dec_out * \
        #          (stdev[:, 0, :].unsqueeze(1).repeat(
        #              1, self.pred_len + self.seq_len, 1))
        #dec_out = dec_out + \
        #          (means[:, 0, :].unsqueeze(1).repeat(
        #              1, self.pred_len + self.seq_len, 1))
        return enc_out


    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc =x_enc.transpose(1,2)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        ## porject back
        #dec_out = self.projection(enc_out)

        ## De-Normalization from Non-stationary Transformer
        #dec_out = dec_out * \
        #          (stdev[:, 0, :].unsqueeze(1).repeat(
        #              1, self.pred_len + self.seq_len, 1))
        #dec_out = dec_out + \
        #          (means[:, 0, :].unsqueeze(1).repeat(
        #              1, self.pred_len + self.seq_len, 1))
        return enc_out

    def forward(self, x_enc, x_mark_enc=None, mask=None):
        if self.task_name == 'forecast' :
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out # [B, L, D]
        if self.task_name == 'anomaly':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        return None
