import pathlib
from typing import cast

import cmi_dss_lib.datamodule.seg
import hydra
import numpy as np
import numpy_utility as npu
import omegaconf
import pandas as pd
import plotly.express as px
from cmi_dss_lib.config import StackingConfig
from cmi_dss_lib.datamodule.stacking import StackingDataModule

project_root_path = pathlib.Path(__file__).parent.parent.parent

overrides_yaml_path = project_root_path / "config" / "exp_for_stacking" / "s1.yaml"
assert overrides_yaml_path.exists()

hydra.initialize(config_path="../conf", version_base="1.2")

cfg = cast(
    StackingConfig,
    hydra.compose("stacking", overrides=list(omegaconf.OmegaConf.load(overrides_yaml_path))),
)
cfg.dir.sub_dir = ".."
cfg.bg_sampling_rate = 0
datamodule = StackingDataModule(cfg)
datamodule.setup("fit")

# i = 10
i = 20

record = datamodule.val_dataloader().dataset[i]
# record = datamodule.train_dataloader().dataset[0]
feature = record["feature"]
assert feature.ndim == 3  # (pred_type, model, duration)

import plotly.express as px

fig = px.imshow(feature, facet_col=0, facet_col_wrap=1, aspect=feature.shape[1] / feature.shape[2])
# set variable back to string
#   https://community.plotly.com/t/cant-set-strings-as-facet-col-in-px-imshow/60904
# for k in range(len(data['variable'])):
#     fig.layout.annotations[k].update(text = data['variable'].values[k])


# update traces to use different coloraxis
for i, t in enumerate(fig.data):
    t.update(coloraxis=f"coloraxis{i+1}")
for fr in fig.frames:
    # update each of the traces in each of the animation frames
    for i, t in enumerate(fr.data):
        t.update(coloraxis=f"coloraxis{i+1}")

# position / config all coloraxis
fig.update_layout(
    coloraxis={"colorbar": {"x": 1, "len": 0.3, "y": 0.85}},
    coloraxis2={
        "colorbar": {
            "x": 1,
            "len": 0.3,
            "y": 0.5,
        },
        "colorscale": "Thermal",
    },
    coloraxis3={
        "colorbar": {"x": 1, "len": 0.3, "y": 0.15},
        "colorscale": "Blues",
    },
)
fig.show()

import matplotlib.pyplot as plt

plt.imshow(feature.swapaxes(0, 2).swapaxes(0, 1), aspect="auto")
plt.show()
