from typing import TypeAlias

from .cnn_spectrogram import CNNSpectrogramConfig
from .lstm_feature_extractor import LSTMFeatureExtractorConfig
from .panns_feature_extractor import PANNsFeatureExtractorConfig
from .spec_feature_extractor import SpecFeatureExtractorConfig

FeatureExtractor: TypeAlias = (
    CNNSpectrogramConfig
    | LSTMFeatureExtractorConfig
    | PANNsFeatureExtractorConfig
    | SpecFeatureExtractorConfig
)
