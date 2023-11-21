from typing import TypeAlias

from .cnn_spectrogram import CNNSpectrogramConfig
from .lstm_feature_extractor import LSTMFeatureExtractorConfig
from .panns_feature_extractor import PANNsFeatureExtractorConfig
from .spec_feature_extractor import SpecFeatureExtractorConfig
from .stacked_attention_lstm_feature_extractor import StackedAttentionLSTMFeatureExtractor
from .times_net_feature_extractor import TimesNetFeatureExtractor

FeatureExtractor: TypeAlias = (
    CNNSpectrogramConfig
    | LSTMFeatureExtractorConfig
    | PANNsFeatureExtractorConfig
    | SpecFeatureExtractorConfig
    | TimesNetFeatureExtractor
    | StackedAttentionLSTMFeatureExtractor
)
