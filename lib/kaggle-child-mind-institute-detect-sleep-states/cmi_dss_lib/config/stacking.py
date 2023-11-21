import dataclasses

from omegaconf import DictConfig

from .dir import DirConfig
from .feature_extractor import FeatureExtractor
from .model import Model
from .train import TrainOptimizerConfig, TrainSchedulerConfig, TrainSplit, TrainSplitType


@dataclasses.dataclass
class StackingConfig(DictConfig):
    dir: DirConfig

    feature_extractor: FeatureExtractor | None

    split: "TrainSplit"
    split_type: "TrainSplitType"

    seed: int
    exp_name: str

    input_model_names: list[str]

    use_amp: bool
    model: Model

    duration: int
    prev_margin_steps: int
    next_margin_steps: int

    downsample_rate: int
    upsample_rate: int

    offset: int
    sigma: int

    bg_sampling_rate: float

    # Training
    epoch: int
    batch_size: int
    num_workers: int
    accelerator: str

    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int

    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int
    val_check_interval: int
    val_after_steps: int

    resume_from_checkpoint: str | None

    early_stopping_patience: int

    optimizer: TrainOptimizerConfig
    scheduler: TrainSchedulerConfig
