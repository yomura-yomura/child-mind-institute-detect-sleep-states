import pathlib

import lightning as L
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..modelmodule.base_chunk import BaseChunkModule
from ..utils.common import trace


def run(cfg: DictConfig, module, datamodule):
    L.seed_everything(cfg.seed)

    with trace("load test dataloader"):
        if cfg.phase == "train":
            datamodule.setup("valid")
            dataloader = datamodule.val_dataloader()
        elif cfg.phase == "test":
            datamodule.setup("test")
            dataloader = datamodule.test_dataloader()
        elif cfg.phase == "dev":
            datamodule.setup("dev")
            dataloader = datamodule.test_dataloader()
        else:
            raise ValueError(f"unexpected {cfg.phase=}")

    pred_dir_path = pathlib.Path(
        cfg.dir.sub_dir,
        "predicted",
        *pathlib.Path(cfg.dir.model_dir).parts[-3:-1],
        cfg.phase
        if cfg.inference_step_offset <= 0
        else f"{cfg.phase}-{cfg.inference_step_offset=}",
        f"{cfg.split.name}",
    )

    pred_dir_path.mkdir(exist_ok=True, parents=True)
    with trace("inference"):
        inference(dataloader, module, cfg.labels, use_amp=cfg.use_amp, pred_dir_path=pred_dir_path)

    if cfg.phase == "train":
        from run.calc_cv import calc_score

        score = calc_score(pred_dir_path, cfg.labels, cfg.downsample_rate)
        print(f"{score:.4f}")
        return score


def inference(
    loader: DataLoader,
    model: L.LightningModule,
    labels: list[str],
    use_amp: bool,
    pred_dir_path: pathlib.Path,
) -> None:
    trainer = L.Trainer(
        devices=1,
        precision=16 if use_amp else 32,
    )
    predictions = trainer.predict(model, loader)

    all_events = ["sleep", "onset", "wakeup"]
    left_events = all_events.copy()
    events = [
        left_events.pop(left_events.index(label[6:] if label.startswith("event_") else label))
        for label in labels
    ]
    for event in left_events:
        events.append(event)

    for series_id, preds in BaseChunkModule._evaluation_epoch_end(
        [pred for preds in predictions for pred in preds]
    ):
        assert preds.shape[-1] == len(labels)

        if len(labels) < 3:
            preds = np.pad(preds, pad_width=[(0, 0), (0, 3 - len(labels))], constant_values=np.nan)
        print(preds.shape)
        preds = preds[:, [events.index(event) for event in all_events]]

        np.savez_compressed(pred_dir_path / f"{series_id}.npz", preds.astype("f2"))
