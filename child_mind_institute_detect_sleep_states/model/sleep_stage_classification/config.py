from typing import TypedDict, Literal

from typing_extensions import Required

__all__ = ["Config", "ModelConfig", "TrainConfig"]


class Config(TypedDict):
    model_architecture: Literal["sleep_stage_classification"]
    model: "ModelConfig"
    # dataset: "DatasetConfig"
    train: "TrainConfig"


class ModelConfig(TypedDict, total=True):
    n_features: int


# class DatasetConfig(TypedDict, total=True):
#     prev_steps_in_epoch: int
#     next_steps_in_epoch: int


class TrainConfig(TypedDict, total=False):
    num_epochs: Required[int]
    train_batch_size: Required[int]
    valid_batch_size: Required[int]

    learning_rate: Required[float]

    weight: float

    n_folds: Required[int]
