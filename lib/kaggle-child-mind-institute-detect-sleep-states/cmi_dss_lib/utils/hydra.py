import os
import sys
from typing import Sequence

import omegaconf


def override_default_hydra_config(
    config_path_or_hydra_arguments: Sequence[str], overrides_dict: dict[str, str] | None = None
) -> None:
    if overrides_dict is None:
        overrides_dict = {}

    for p in config_path_or_hydra_arguments:
        if os.path.exists(p):
            for k, v in (item.split("=", maxsplit=1) for item in omegaconf.OmegaConf.load(p)):
                if k in overrides_dict.keys():
                    print(f"Info: {k}={overrides_dict[k]} is replaced with {k}={v}")
                overrides_dict[k] = v
        else:
            k, v = p.split("=", maxsplit=1)
            if k in overrides_dict.keys():
                print(f"Info: {k}={overrides_dict[k]} is replaced with {k}={v}")
            overrides_dict[k] = v
    sys.argv = sys.argv[:1] + [f"{k}={v}" for k, v in overrides_dict.items()]
