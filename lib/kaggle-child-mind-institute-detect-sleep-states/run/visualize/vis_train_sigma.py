import pathlib
from typing import cast

import cmi_dss_lib.datamodule.seg
import hydra
import omegaconf
from cmi_dss_lib.config import TrainConfig

project_root_path = pathlib.Path(__file__).parent.parent.parent

exp_name = "ranchantan/exp050-transformer-decoder_retry_resume"
i_fold = 0

hydra.initialize(config_path="../conf", version_base="1.2")

overrides_yaml_path = (
    project_root_path
    / "cmi-dss-ensemble-models"
    / exp_name
    / f"fold_{i_fold + 1}"
    / ".hydra"
    / "overrides.yaml"
)
assert overrides_yaml_path.exists()

cfg = cast(
    TrainConfig,
    hydra.compose("train", overrides=list(omegaconf.OmegaConf.load(overrides_yaml_path))),
)

cfg.sigma = 12 * 5
cfg.offset = 12 * 20
datamodule = cmi_dss_lib.datamodule.seg.SegDataModule(cfg)
datamodule.setup("fit")
val_dataset = datamodule.train_dataloader().dataset

import numpy as np
import plotly.express as px

while True:
    feat_record = val_dataset[0]
    i = np.argmax(feat_record["label"][:, 1])
    if i > 0:
        break


labels_with_sigma = feat_record["label"][:, 1][i - cfg.offset * 2 : i + cfg.offset * 2].numpy()
x = np.arange(-len(labels_with_sigma) // 2, len(labels_with_sigma) // 2)
fig = px.line(title=f"{cfg.sigma=}, {cfg.offset=}", x=x, y=labels_with_sigma)
fig.show()

import standard_fit.plotly.express as sfpx

sfpx.fit(fig, "gaussian")
