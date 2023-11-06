from typing import TypeAlias

from .spec_1d import Spec1DConfig
from .spec_2d_cnn import Spec2DCNNConfig

Model: TypeAlias = Spec1DConfig | Spec2DCNNConfig
