import dataclasses
from typing import Literal


@dataclasses.dataclass
class Spec2DCNNConfig:
    name: Literal["Spec2DCNN"]
    segmentation_model_name: Literal["unet", "manet"]
    encoder_name: Literal["resnet16", "resnet34"]
    encoder_weights: Literal["imagenet", None]
