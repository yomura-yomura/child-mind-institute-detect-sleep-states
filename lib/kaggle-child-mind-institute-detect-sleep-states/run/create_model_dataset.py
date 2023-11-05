import pathlib

import hydra
import torch
import tqdm
from cmi_dss_lib.modelmodule.seg import SegModel
from omegaconf import DictConfig

project_root_path = pathlib.Path(__file__).parent.parent


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    train_output_dir_path = project_root_path / "output"
    dataset_output_dir_path = project_root_path / "output_dataset"
    print(train_output_dir_path.resolve())

    for p in tqdm.tqdm(sorted(train_output_dir_path.glob("train/exp003-fold*/single/best.ckpt"))):
        module = SegModel.load_from_checkpoint(
            f"{p}",
            cfg=cfg,
            val_event_df=None,
            feature_dim=len(cfg.features),
            num_classes=len(cfg.labels),
            duration=cfg.duration,
        )

        output_dir_path = dataset_output_dir_path / p.relative_to(train_output_dir_path).parent
        output_dir_path.mkdir(exist_ok=True, parents=True)
        torch.save(
            module.model.state_dict(),
            output_dir_path / "best_model.pth",
        )


if __name__ == "__main__":
    main()
