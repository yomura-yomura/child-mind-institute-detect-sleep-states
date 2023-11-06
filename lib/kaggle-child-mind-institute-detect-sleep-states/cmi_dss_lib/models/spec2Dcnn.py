from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from cmi_dss_lib.augmentation.cutmix import Cutmix
from cmi_dss_lib.augmentation.mixup import Mixup


class Spec2DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.decoder = decoder
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
            do_mixup: do mixup augmentation
            do_cutmix: do cutmix augmentation
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_time_steps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_time_steps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_time_steps)
        logits = self.decoder(x)  # (batch_size, n_classes, n_time_steps)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss

        return output


class Spec1DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.decoder = decoder
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
            do_mixup: do mixup augmentation
            do_cutmix: do cutmix augmentation
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_time_steps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_time_steps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_time_steps)
        logits = self.decoder(x)  # (batch_size, n_classes, n_time_steps)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss

        return output