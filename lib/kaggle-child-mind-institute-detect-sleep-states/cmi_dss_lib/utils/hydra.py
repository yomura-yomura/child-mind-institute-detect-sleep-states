import os
import sys
from typing import Sequence

import omegaconf


def override_default_hydra_config(
    config_path_or_hydra_arguments: Sequence[str], overrides_dict: dict[str, str] | None = None
) -> None:
    overrides_dict_ = {}

    for p in config_path_or_hydra_arguments:
        if os.path.exists(p):
            for k, v in (item.split("=", maxsplit=1) for item in omegaconf.OmegaConf.load(p)):
                if k in overrides_dict_.keys():
                    print(f"Info: {k}={overrides_dict_[k]} is replaced with {k}={v}")
                overrides_dict_[k] = v
        else:
            try:
                k, v = p.split("=", maxsplit=1)
            except ValueError:
                raise ValueError(f"unexpected config_path_or_hydra_arguments: {p}")
            if k in overrides_dict_.keys():
                print(f"Info: {k}={overrides_dict_[k]} is replaced with {k}={v}")
            overrides_dict_[k] = v

    for k, v in overrides_dict.items():
        if k in overrides_dict_.keys():
            print(f"Info: {k}={overrides_dict_[k]} is replaced with {k}={v}")
        overrides_dict_[k] = v
    sys.argv = sys.argv[:1] + [f"{k}={v}" for k, v in overrides_dict_.items()]
