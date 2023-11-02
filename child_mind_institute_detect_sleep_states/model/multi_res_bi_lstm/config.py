from typing import Literal, TypedDict

from typing_extensions import Required

__all__ = ["ModelConfig", "TrainConfig"]


class Config(TypedDict):
    model_architecture: Literal["multi_res_bi_lstm"]
    model: "ModelConfig"
    train: "TrainConfig"


class ModelConfig(TypedDict, total=True):
    input_size: int
    hidden_size: int
    out_size: int
    n_layers: int

    max_chunk_size: int


class TrainConfig(TypedDict, total=False):
    num_epochs: Required[int]
    train_batch_size: Required[int]
    valid_batch_size: Required[int]

    optimizer: Required["TrainOptimizerConfig"]

    n_folds: Required[int]


class TrainOptimizerConfig(TypedDict, total=False):
    learning_rate: Required[float]
    weight_decay: float
    scheduler: "TrainOptimizerSchedulerConfig"


class TrainOptimizerSchedulerConfig(TypedDict, total=False):
    eta_min: float
    last_epoch: int
