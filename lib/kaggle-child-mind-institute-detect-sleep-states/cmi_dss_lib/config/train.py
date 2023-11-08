import dataclasses
from typing import Literal

from .decoder import Decoder
from .dir import DirConfig
from .feature_extractor import FeatureExtractor
from .model import Model

__all__ = [
    "TrainConfig",
    "TrainSchedulerConfig",
    "TrainAugmentationConfig",
    "TrainPostProcessAugmentationConfig",
    "TrainOptimizerConfig",
]


@dataclasses.dataclass
class TrainConfig:
    dir: DirConfig

    model_dim: Literal[1, 2]
    model: Model
    feature_extractor: FeatureExtractor
    decoder: Decoder

    split: str  # TODO: should be removed

    seed: int
    exp_name: str
    phase: Literal["train", "test"]

    # weight

    duration: int
    downsample_rate: int
    upsample_rate: int

    epoch: int
    batch_size: int
    num_workers: int
    accelerator: Literal["auto"]
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: Literal["EventDetectionAP"]
    monitor_mode: Literal["min", "max"]
    check_val_every_n_epoch: int

    offset: int
    sigma: int
    bg_sampling_rate: float

    augmentation: "TrainAugmentationConfig"

    post_process: "TrainPostProcessAugmentationConfig"

    labels: list[Literal["awake", "event_onset", "event_wakeup"]]

    features: list[
        Literal[
            "anglez",
            "enmo",
            "month_sin",
            "month_cos",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
        ]
    ]

    optimizer: "TrainOptimizerConfig"

    scheduler: "TrainSchedulerConfig"

    early_stopping_patience: int


@dataclasses.dataclass
class TrainAugmentationConfig:
    mixup_prob: float
    mixup_alpha: float
    cutmix_prob: float
    cutmix_alpha: float


@dataclasses.dataclass
class TrainPostProcessAugmentationConfig:
    score_th: float
    distance: int


@dataclasses.dataclass
class TrainOptimizerConfig:
    lr: float


@dataclasses.dataclass
class TrainSchedulerConfig:
    num_warmup_steps: int
