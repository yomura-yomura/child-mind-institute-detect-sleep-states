from typing import Optional

import cmi_dss_lib.models.decoder.unet1ddecoder
import cmi_dss_lib.models.encoder.resnet_1d
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
        model_dim: int,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        # self.encoder = cmi_dss_lib.models.decoder.unet1ddecoder.UNet1DDecoder(
        #     n_channels=feature_extractor.height,
        #     n_classes=feature_extractor.height,
        #     duration=num_time_steps,
        #     bilinear=False,
        # )
        block_dim = list(
            zip(
                [64, 64, 128, 128, 256, 256, 512, 512],
                [
                    num_time_steps,
                    num_time_steps,
                    num_time_steps // 2,
                    num_time_steps // 2,
                    num_time_steps // 4,
                    num_time_steps // 4,
                    num_time_steps // 8,
                    num_time_steps // 8,
                ],
            )
        )
        self.channels_fc = nn.Linear(block_dim[-1][1], num_time_steps)
        self.encoder = cmi_dss_lib.models.encoder.resnet_1d.ResNet1d(
            input_dim=(feature_extractor.height, num_time_steps),
            blocks_dim=block_dim,
            n_classes=feature_extractor.height,
            # kernel_size=17,
            kernel_size=3,
            dropout_rate=0,
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

        # pool over n_channels dimension
        if self.model_dim == 1:
            # x = x.transpose(2, 3)  # (batch_size, n_time_steps, height)
            # x = self.channels_fc(x)  # (batch_size, n_time_steps, 1)
            # x = x.squeeze(-1)  # (batch_size, n_channels, n_time_steps)
            x = self.encoder(x)  # (batch_size, n_time_steps, height)
            x = x.transpose(1, 2)
            x = self.channels_fc(x)
            # print(x.shape)
            # x = x.transpose(1, 2)
            logits = self.decoder(x)  # (batch_size, n_classes, n_time_steps)
        elif self.model_dim == 2:
            # pool over n_channels dimension
            x = x.transpose(1, 3)  # (batch_size, n_time_steps, height, n_channels)
            x = self.channels_fc(x)  # (batch_size, n_time_steps, height, 1)
            x = x.squeeze(-1)  # (batch_size, n_time_steps, height)
            x = x.transpose(1, 2)  # (batch_size, height, n_time_steps)
            x = self.encoder(x)  # (batch_size, n_time_steps, height)
            # x = x.transpose(1, 2)
            logits = self.decoder(x)  # (batch_size, n_classes, n_time_steps)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss

        return output
