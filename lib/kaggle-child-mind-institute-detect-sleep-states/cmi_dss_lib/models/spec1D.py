from typing import Optional

import cmi_dss_lib.models.decoder.unet1ddecoder
import torch
import torch.nn as nn
from cmi_dss_lib.augmentation.cutmix import Cutmix
from cmi_dss_lib.augmentation.mixup import Mixup


class Spec1D(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        num_time_steps: int,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.channels_fc = nn.Linear(feature_extractor.height, 1)
        self.encoder = cmi_dss_lib.models.decoder.unet1ddecoder.UNet1DDecoder(
            n_channels=1,
            n_classes=1,
            duration=num_time_steps,
            bilinear=False,
        )
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
            x (torch.Tensor): (batch_size, n_channels, n_time_steps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_time_steps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_time_steps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_time_steps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        # pool over n_channels dimension
        x = x.transpose(2, 3)  # (batch_size, n_channels, n_time_steps, height)
        x = self.channels_fc(x)  # (batch_size, n_channels, n_time_steps, 1)
        x = x.squeeze(-1)  # (batch_size, n_channels, n_time_steps)
        x = self.encoder(x)  # (batch_size, n_time_steps, height)
        # print(x.shape)
        x = x.transpose(1, 2)
        logits = self.decoder(x)  # (batch_size, n_classes, n_time_steps)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss

        return output
