from typing import Union

import torch.nn as nn
from cmi_dss_lib.models.decoder.lstmdecoder import LSTMDecoder
from cmi_dss_lib.models.decoder.mlpdecoder import MLPDecoder
from cmi_dss_lib.models.decoder.transformerdecoder import TransformerDecoder
from cmi_dss_lib.models.decoder.unet1ddecoder import UNet1DDecoder
from cmi_dss_lib.models.feature_extractor.cnn import CNNSpectrogram
from cmi_dss_lib.models.feature_extractor.lstm import LSTMFeatureExtractor
from cmi_dss_lib.models.feature_extractor.panns import PANNsFeatureExtractor
from cmi_dss_lib.models.feature_extractor.spectrogram import SpecFeatureExtractor
from cmi_dss_lib.models.feature_extractor.stacked_gru import StackedGRUFeatureExtractor
from cmi_dss_lib.models.feature_extractor.stacked_lstm import StackedLSTMFeatureExtractor
from cmi_dss_lib.models.feature_extractor.timesnet import TimesNetFeatureExtractor
from cmi_dss_lib.models.spec1D import Spec1D
from cmi_dss_lib.models.spec2Dcnn import Spec2DCNN

from child_mind_institute_detect_sleep_states.model.multi_res_bi_lstm_attention.model import (
    FOGModel,
)

from ..config import TrainConfig

FEATURE_EXTRACTORS = Union[
    CNNSpectrogram,
    PANNsFeatureExtractor,
    LSTMFeatureExtractor,
    SpecFeatureExtractor,
    StackedGRUFeatureExtractor,
    StackedLSTMFeatureExtractor,
    TimesNetFeatureExtractor,
    FOGModel,
]
DECODERS = Union[UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder]
MODELS = Union[Spec1D, Spec2DCNN]


def get_feature_extractor(
    cfg: TrainConfig, feature_dim: int, num_time_steps: int
) -> FEATURE_EXTRACTORS:
    feature_extractor: FEATURE_EXTRACTORS
    if cfg.feature_extractor.name == "CNNSpectrogram":
        assert cfg.model_dim == 2
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim,
            base_filters=cfg.feature_extractor.base_filters,
            kernel_sizes=cfg.feature_extractor.kernel_sizes,
            stride=cfg.feature_extractor.stride,
            sigmoid=cfg.feature_extractor.sigmoid,
            output_size=num_time_steps,
            conv=nn.Conv1d,
            reinit=cfg.feature_extractor.reinit,
        )
    elif cfg.feature_extractor.name == "PANNsFeatureExtractor":
        assert cfg.model_dim == 2
        feature_extractor = PANNsFeatureExtractor(
            in_channels=feature_dim,
            base_filters=cfg.feature_extractor.base_filters,
            kernel_sizes=cfg.feature_extractor.kernel_sizes,
            stride=cfg.feature_extractor.stride,
            sigmoid=cfg.feature_extractor.sigmoid,
            output_size=num_time_steps,
            conv=nn.Conv1d,
            reinit=cfg.feature_extractor.reinit,
            win_length=cfg.feature_extractor.win_length,
        )
    elif cfg.feature_extractor.name == "LSTMFeatureExtractor":
        feature_extractor = LSTMFeatureExtractor(
            in_channels=feature_dim,
            hidden_size=cfg.feature_extractor.hidden_size,
            num_layers=cfg.feature_extractor.num_layers,
            bidirectional=cfg.feature_extractor.bidirectional,
            model_dim=cfg.model_dim,
            out_size=num_time_steps,
        )
    elif cfg.feature_extractor.name == "SpecFeatureExtractor":
        assert cfg.model_dim == 2
        feature_extractor = SpecFeatureExtractor(
            in_channels=feature_dim,
            height=cfg.feature_extractor.height,
            hop_length=cfg.feature_extractor.hop_length,
            win_length=cfg.feature_extractor.win_length,
            out_size=num_time_steps,
        )

    elif cfg.feature_extractor.name == "StackedGRUFeatureExtractor":
        assert cfg.model_dim == 2
        feature_extractor = StackedGRUFeatureExtractor(
            in_channels=feature_dim,
            hidden_size=cfg.feature_extractor.hidden_size,
            num_layers=cfg.feature_extractor.num_layers,
            bidirectional=cfg.feature_extractor.bidirectional,
            out_size=num_time_steps,
        )

    elif cfg.feature_extractor.name == "StackedLSTMFeatureExtractor":
        assert cfg.model_dim == 2
        feature_extractor = StackedLSTMFeatureExtractor(
            in_channels=feature_dim,
            hidden_size=cfg.feature_extractor.hidden_size,
            num_layers=cfg.feature_extractor.num_layers,
            bidirectional=cfg.feature_extractor.bidirectional,
            out_size=num_time_steps,
        )
    elif cfg.feature_extractor.name == "TimesNetFeatureExtractor":
        feature_extractor = TimesNetFeatureExtractor(
            in_channels=feature_dim,
            height=cfg.feature_extractor.height,
            dim_model=cfg.feature_extractor.dim_model,
            encoder_layers=cfg.feature_extractor.encoder_layers,
            times_blocks=cfg.feature_extractor.times_blocks,
            num_kernels=cfg.feature_extractor.num_kernels,
            dropout=cfg.feature_extractor.dropout,
            dim_fc=cfg.feature_extractor.dim_fc,
            embed_encoding=cfg.feature_extractor.embed_encoding,
            freq=cfg.feature_extractor.freq,
            task=cfg.feature_extractor.task,
            is_fc=cfg.feature_extractor.is_fc,
            out_size=num_time_steps,
        )
    elif cfg.feature_extractor.name == "StackedAttentionLSTMFeatureExtractor":
        feature_extractor = FOGModel(
            patch_size=cfg.feature_extractor.patch_size,
            duration=cfg.duration,
            n_features=feature_dim,
            n_encoder_layers=cfg.feature_extractor.n_encoder_layers,
            n_lstm_layers=cfg.feature_extractor.n_lstm_layers,
            out_size=num_time_steps,
            dropout=cfg.feature_extractor.dropout,
            mha_embed_dim=cfg.feature_extractor.mha_embed_dim,
            mha_n_heads=cfg.feature_extractor.mha_n_heads,
            mha_dropout=cfg.feature_extractor.mha_dropout,
        )
        feature_extractor.height = cfg.feature_extractor.mha_embed_dim
        feature_extractor.out_chans = 1
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.feature_extractor.name}")

    return feature_extractor


def get_decoder(
    cfg: TrainConfig, n_channels: int, n_classes: int, num_time_steps: int
) -> DECODERS:
    decoder: DECODERS
    if cfg.decoder.name == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_time_steps,
            bilinear=cfg.decoder.bilinear,
            se=cfg.decoder.se,
            res=cfg.decoder.res,
            scale_factor=cfg.decoder.scale_factor,
            dropout=cfg.decoder.dropout,
        )
    elif cfg.decoder.name == "LSTMDecoder":
        decoder = LSTMDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            bidirectional=cfg.decoder.bidirectional,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "TransformerDecoder":
        decoder = TransformerDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            nhead=cfg.decoder.nhead,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "MLPDecoder":
        decoder = MLPDecoder(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Invalid decoder name: {cfg.decoder.name}")

    return decoder


def get_model(cfg: TrainConfig, feature_dim: int, n_classes: int, num_time_steps: int) -> MODELS:
    model: MODELS
    feature_extractor = get_feature_extractor(cfg, feature_dim, num_time_steps)
    decoder = get_decoder(cfg, feature_extractor.height, n_classes, num_time_steps)
    if cfg.model.name == "Spec2DCNN":
        assert cfg.model_dim == 2

        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            segmentation_model_name=cfg.model.segmentation_model_name,
            decoder=decoder,
            encoder_name=cfg.model.encoder_name,
            in_channels=feature_extractor.out_chans,
            encoder_weights=cfg.model.encoder_weights,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )
    elif cfg.model.name == "Spec1D":
        model = Spec1D(
            feature_extractor=feature_extractor,
            decoder=decoder,
            encoder_name=cfg.model.encoder_name,
            num_time_steps=num_time_steps,
            model_dim=cfg.model_dim,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )
    else:
        raise NotImplementedError

    return model
