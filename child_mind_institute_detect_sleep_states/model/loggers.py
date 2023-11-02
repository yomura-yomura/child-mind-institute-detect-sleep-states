import pathlib
from typing import Any, Literal, Optional, Union

import wandb
from lightning.pytorch.loggers import WandbLogger

__all__ = ["WandbLogger", "get_versioning_wandb_group_name"]


def get_versioning_wandb_group_name(wandb_group_name: str) -> str:
    api = wandb.Api()
    possible_exp_names = set(
        run._attrs["group"]
        for run in api.runs(
            "ranchan/child-mind-institute-detect-sleep-states",
            filters={"group": {"$regex": rf"^{wandb_group_name}(_v\d+)?$"}},
        )
    )

    if len(possible_exp_names) == 0:
        target_version = 0
    else:
        target_version = 1

    possible_exp_names -= {wandb_group_name}
    if len(possible_exp_names) > 0:
        target_version += max(int(name[len(f"{wandb_group_name}_v") :]) for name in possible_exp_names)

    if target_version > 0:
        wandb_group_name = f"{wandb_group_name}_v{target_version}"
    return wandb_group_name
