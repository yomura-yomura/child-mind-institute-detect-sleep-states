import dataclasses
from typing import Literal

from omegaconf import DictConfig

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
    "PsuedoLabelConfig"
]


@dataclasses.dataclass
class TrainConfig(DictConfig):
    dir: DirConfig

    model_dim: Literal[1, 2]
    model: Model
    feature_extractor: FeatureExtractor
    decoder: Decoder

    split: "TrainSplit"
    split_type: "TrainSplitType"

    seed: int
    exp_name: str

    phase: Literal["train", "test"]
    scale_type: Literal["constant", "robust_scaler"]

    # psuedo labeling
    psuedo_label:"PsuedoLabelConfig"

    # weight

    duration: int
    prev_margin_steps: int
    next_margin_steps: int

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

    check_val_every_n_epoch: int | None
    val_check_interval: float | int | None
    val_after_steps: int

    offset: int
    sigma: int
    bg_sampling_rate: float

    augmentation: "TrainAugmentationConfig"

    post_process: "TrainPostProcessAugmentationConfig"

    labels: list[Literal["sleep", "event_onset", "event_wakeup"]]

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

    resume_from_checkpoint: str | None



@dataclasses.dataclass
class PsuedoLabelConfig:
    use_psuedo: bool
    save_psuedo: bool
    save_path:str
    use_version:int
    v0:"PsuedoLabelv0Config"
    v1:"PsuedoLabelv1Config"
@dataclasses.dataclass
class PsuedoLabelv0Config:
    path_psuedo:str
    th_sleep:float
    th_prop:float

class PsuedoLabelv1Config:
    path_psuedo:str
    watch_interval:float
@dataclasses.dataclass
class TrainSplit(DictConfig):
    name: Literal["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]


@dataclasses.dataclass
class TrainSplitType(DictConfig):
    name: Literal["08_nan", "08_nan_06_repeat_rate"]


@dataclasses.dataclass
class TrainAugmentationConfig(DictConfig):
    mixup_prob: float
    mixup_alpha: float
    cutmix_prob: float
    cutmix_alpha: float


@dataclasses.dataclass
class TrainPostProcessAugmentationConfig(DictConfig):
    score_th: float
    distance: int


@dataclasses.dataclass
class TrainOptimizerConfig(DictConfig):
    lr: float


@dataclasses.dataclass
class TrainSchedulerConfig(DictConfig):
    num_warmup_steps: int
