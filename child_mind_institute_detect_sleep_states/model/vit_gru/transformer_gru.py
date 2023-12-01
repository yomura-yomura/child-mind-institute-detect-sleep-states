import datetime
import inspect
import json

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_relpos import VisionTransformerRelPos

torch.manual_seed(datetime.datetime.now().microsecond)


# the core xformer/rnn backbone;
class VitGru(nn.Module):
    def __init__(
        self,
        # params,
        patch,
        seq,
        dims=256,
        nheads=8,
        act_layer="GELU",
        dropout=0.2,
        rnn="GRU",
        rnn_layers=1,
        final_mult=4,
        patch_dropout=0.0,
        patch_act="Identity",
        # seg=False,
        pre_norm=False,
        patch_norm=False,
        xformer_layers=2,
        xformer_init_1=0.1,
        xformer_init_2=0.1,
        xformer_init_scale=1.0,
        xformer_attn_drop_rate=0.1,
        xformer_drop_path_rate=0.1,
        deberta=False,
        h0=False,
        rel_pos="bias",
        alibi=True,
    ):
        super().__init__()
        self.deberta = deberta
        self.ch = 5  # feature_num
        # self.seg = seg
        self.seq = seq
        self.patch = patch
        self.pre_norm = pre_norm
        self.patch_norm = patch_norm
        if not deberta:
            self.xformer = (
                (VisionTransformerRelPos if rel_pos else VisionTransformer)(
                    img_size=(seq, 1),
                    patch_size=(1, 1),
                    in_chans=dims,
                    num_classes=0,
                    global_pool="",
                    embed_dim=dims,
                    num_heads=nheads,
                    embed_layer=IdentityPatch,
                    act_layer=getattr(nn, act_layer),
                    depth=xformer_layers,
                    init_values=xformer_init_1,
                    class_token=False,
                    drop_rate=dropout,
                    attn_drop_rate=xformer_attn_drop_rate,
                    drop_path_rate=xformer_drop_path_rate,
                    **({"rel_pos_type": rel_pos} if rel_pos else {}),
                )
                if xformer_layers > 0
                else None
            )
            if xformer_layers > 0 and rel_pos == "bias" and alibi > 0:
                for i, b in enumerate(self.xformer.blocks):
                    b.attn.rel_pos.relative_position_bias_table.data = alibi_bias(
                        b.attn.rel_pos.relative_position_bias_table.data, alibi
                    )
            if xformer_layers > 0 and not rel_pos:
                self.xformer.pos_embed.data /= 2
                self.xformer.pos_embed.data[:] += 0.02 * getPositionalEncoding(seq, dims, 1000).unsqueeze(0)
            if xformer_layers > 0:
                for i, b in enumerate(self.xformer.blocks):
                    b.ls1.gamma.data[:] = torch.tensor(xformer_init_1 * xformer_init_scale**i)
                    b.ls2.gamma.data[:] = torch.tensor(xformer_init_2 * xformer_init_scale**i)
        else:
            # deberta v3

            # config = transformers.AutoConfig.from_pretrained('microsoft/deberta-v3-xsmall')
            cjson = '{"return_dict": true, "output_hidden_states": false, "output_attentions": false, "torchscript": false, "torch_dtype": null, "use_bfloat16": false, "tf_legacy_loss": false, "pruned_heads": {}, "tie_word_embeddings": true, "is_encoder_decoder": false, "is_decoder": false, "cross_attention_hidden_size": null, "add_cross_attention": false, "tie_encoder_decoder": false, "max_length": 20, "min_length": 0, "do_sample": false, "early_stopping": false, "num_beams": 1, "num_beam_groups": 1, "diversity_penalty": 0.0, "temperature": 1.0, "top_k": 50, "top_p": 1.0, "typical_p": 1.0, "repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0, "bad_words_ids": null, "num_return_sequences": 1, "chunk_size_feed_forward": 0, "output_scores": false, "return_dict_in_generate": false, "forced_bos_token_id": null, "forced_eos_token_id": null, "remove_invalid_values": false, "exponential_decay_length_penalty": null, "suppress_tokens": null, "begin_suppress_tokens": null, "architectures": null, "finetuning_task": null, "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "label2id": {"LABEL_0": 0, "LABEL_1": 1}, "tokenizer_class": null, "prefix": null, "bos_token_id": null, "pad_token_id": 0, "eos_token_id": null, "sep_token_id": null, "decoder_start_token_id": null, "task_specific_params": null, "problem_type": null, "_name_or_path": "microsoft/deberta-v3-xsmall", "transformers_version": "4.27.1", "model_type": "deberta-v2", "position_buckets": 256, "norm_rel_ebd": "layer_norm", "share_att_key": true, "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6, "intermediate_size": 1536, "hidden_act": "gelu", "hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1, "max_position_embeddings": 512, "type_vocab_size": 0, "initializer_range": 0.02, "relative_attention": true, "max_relative_positions": -1, "position_biased_input": false, "pos_att_type": ["p2c", "c2p"], "vocab_size": 128100, "layer_norm_eps": 1e-07, "pooler_hidden_size": 384, "pooler_dropout": 0, "pooler_hidden_act": "gelu"}'
            config = transformers.DebertaV2Config.from_dict(json.loads(cjson))
            config.hidden_size = dims
            config.intermediate_size = dims * 4
            config.num_attention_heads = nheads
            config.num_hidden_layers = xformer_layers
            config.attention_probs_dropout_prob = xformer_attn_drop_rate
            config.hidden_dropout_prob = dropout
            self.xformer = transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder(config)
            for i, b in enumerate(self.xformer.layer):
                b.attention.output.dense.weight.data *= xformer_init_1 * xformer_init_scale**i
                b.intermediate.dense.weight.data *= xformer_init_2 * xformer_init_scale**i

        # if xformer_layers == 0 or deberta:
        # if seg:
        #     self.embed = nn.Sequential(
        #         SegmentationModel(**getParams(SegmentationModel, params)),
        #         nn.GroupNorm(4, dims),
        #     )
        # else:
        self.embed = nn.Sequential(
            PatchEmbed(
                img_size=(seq, 3), patch_size=(1, 3), in_chans=self.ch, embed_dim=dims, bias=not self.patch_norm
            ),
            GroupNorm1d(4, dims) if self.patch_norm else nn.Identity(),
        )
        self.norm = nn.LayerNorm(dims)
        self.patch_act = getattr(nn, patch_act)()
        self.dropout = nn.Dropout(dropout)
        self.patch_dropout = nn.Dropout1d(patch_dropout)
        self.h0 = nn.Parameter(h0 * torch.randn(2 * rnn_layers, dims // 2 * final_mult)) if h0 else None
        self.rnn = (
            getattr(nn, rnn)(
                dims,
                dims // 2 * final_mult,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=True,
                batch_first=True,
            )
            if rnn is not None and rnn_layers > 0
            else None
        )

    def forward(self, x):
        # x: (batch_size, in_channels, time_steps)
        patch = self.patch
        x = x.reshape(x.shape[0], x.shape[1], self.seq // patch, patch)
        # x: (batch_size, in_channels, ,patch)

        attn = x.reshape(x.shape[0], -1, patch * 3)
        attn = 1 * (attn.std(-1) > 1e-5)
        x = x.permute(0, 2, 1, 3)

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.embed(x)
        x = self.patch_act(x)
        if self.seg:
            x = x.mean(-1).permute(0, 2, 1)
        if self.pre_norm:
            x = self.norm(x)
        x = x * attn.unsqueeze(-1)
        x = self.xformer(x, attn)["last_hidden_state"] if self.deberta else (self.xformer(x) if self.xformer else x)
        x = x * attn.unsqueeze(-1)
        if self.rnn is not None:
            x = self.dropout(x)
            x = self.patch_dropout(x)
        xt = x
        x = (
            self.rnn(x, self.h0.unsqueeze(1).repeat(1, x.shape[0], 1) if torch.is_tensor(self.h0) else self.h0)[0]
            if self.rnn is not None
            else x
        )
        if self.xformer is None:
            xt = x

        return x, xt


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


# seg model;
class SegmentationModel(nn.Module):
    def __init__(
        self,
        dims,
        encoder=  # shallow is better, mobilevit it is;
        #  'tu-tf_efficientnetv2_b3', #  'tu-tf_efficientnet_b3',
        "tu-mobilevitv2_075",
        encoder_depth=4,
        dropout=0.2,
        stack=False,
    ):
        super().__init__()
        self.stack = stack
        self.mult = 2 ** (encoder_depth + 1)
        self.seg_model = smp.Unet(
            encoder,
            in_channels=1 if stack else 3,
            encoder_depth=encoder_depth,
            decoder_channels=[256, 128, 64, 32, 16][:encoder_depth],
            classes=dims,
            **({"encoder_weights": None} if OFFLINE else {}),
        )
        # iterate over model, activate xformer dropout
        for name, module in self.seg_model.named_modules():
            if isinstance(module, nn.Dropout) and "attn_drop" not in name:
                module.p = dropout
        # thin layers;
        self.seg_model.encoder.model.stages_2[1].transformer = nn.Sequential(
            self.seg_model.encoder.model.stages_2[1].transformer[:2],
        )

    def forward(self, x):
        if self.stack:
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.shape[0], -1, x.shape[-1]).unsqueeze(1).permute(0, 1, 3, 2)
        else:
            x = x.permute(0, 3, 2, 1)

        xo = x
        x = F.pad(
            x,
            (
                0,
                self.mult - x.shape[-1] % self.mult,
            ),
        )
        x = self.seg_model(x)
        x = x[:, :, :, : xo.shape[-1]]
        re_expand = xo.shape[-2] // x.shape[-2]
        if re_expand > 0:
            x = F.interpolate(x, scale_factor=(re_expand, 1), mode="bilinear")
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
