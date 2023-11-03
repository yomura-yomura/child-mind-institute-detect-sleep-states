from typing import Literal, TypeAlias, TypedDict

from typing_extensions import Required

from ..dataset import FeatureNames

__all__ = [
    "Config",
    "ModelConfig",
    "DatasetConfig",
    "TrainConfig",
    "EvalConfig",
    "ModelArchitecture",
    "TrainFoldType",
]


ModelArchitecture: TypeAlias = Literal["multi_res_bi_gru"]


class Config(TypedDict):
    model_architecture: ModelArchitecture
    exp_name: str

    model: "ModelConfig"
    dataset: "DatasetConfig"
    train: "TrainConfig"
    eval: "EvalConfig"


class ModelConfig(TypedDict, total=True):
    hidden_size: int
    out_size: int
    n_layers: int

    max_chunk_size: int


TrainDatasetType: TypeAlias = Literal["base", "with_part_id"]


class DatasetConfig(TypedDict):
    sigma: int
    agg_interval: int
    features: FeatureNames
    train_dataset_type: TrainDatasetType


TrainFoldType: TypeAlias = Literal["group", "stratified_group"]


class TrainConfig(TypedDict, total=False):
    num_epochs: Required[int]
    train_batch_size: Required[int]
    valid_batch_size: Required[int]

    learning_rate: Required[int]

    fold_type: Required[TrainFoldType]
    n_folds: Required[int]

    early_stopping_patience: int
    early_stopping_max_epoch: int

    save_top_k_models: int

    optimizer: Required["TrainOptimizerConfig"]


class TrainOptimizerConfig(TypedDict, total=False):
    weight_decay: float
    scheduler: "TrainOptimizerSchedulerConfig"


class TrainOptimizerSchedulerConfig(TypedDict, total=False):
    eta_min: float
    last_epoch: int


class EvalConfig(TypedDict, total=False):
    window: int
