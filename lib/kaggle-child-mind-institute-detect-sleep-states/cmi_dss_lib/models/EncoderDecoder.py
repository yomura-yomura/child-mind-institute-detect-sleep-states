from typing import Optional

import cmi_dss_lib.models.decoder.unet1ddecoder
import cmi_dss_lib.models.encoder.resnet_1d
import torch
import torch.nn as nn
from cmi_dss_lib.augmentation.cutmix import Cutmix
from cmi_dss_lib.augmentation.mixup import Mixup


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        num_time_steps: int,
        model_dim: int,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.feature_extractor = feature_extractor
        self.decoder = decoder

        self.channels_fc = nn.Linear(block_dim[-1][1], num_time_steps)
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor):
                1D: (batch_size, height, n_time_steps)
                2D: (batch_size, n_channels, height, n_time_steps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_time_steps, n_classes)
            do_cutmix:
            do_mixup:
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_time_steps, n_classes)

        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_time_steps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = x.transpose(1, 3)  # (batch_size, n_time_steps, height, n_channels)
        x = x.squeeze(-1)  # (batch_size, n_time_steps, height)
        
        x = x.transpose(1, 2)  # (batch_size, height, n_time_steps)

        logits = self.decoder(x)  # (batch_size, n_classes, n_time_steps)
        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss

        return output
