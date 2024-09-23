import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .swin import SwinTransformerV2


class BaseModel(nn.Module):
    '''Base model.'''

    def __init__(self, backbone: str) -> None:
        '''Base model.

        Args:
            backbone (str): Backbone to use.

        Raises:
            ValueError: Whether given backbone name is valid.
        '''
        super().__init__()

        match backbone:
            case 'swin':
                self.backbone = SwinTransformerV2()
            case _:
                raise ValueError(f'Backbone {backbone} is invalid.')

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Linear(self.backbone.size_out, 1)
        )
        self.head.apply(self._init_weight)

    def _init_weight(self, m: nn.Module) -> None:
        '''Initialize weights of layers.

        Args:
            m (torch.nn.Module): Layers to initialize.
        '''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_channel, height,
            width).

        Returns:
            torch.Tensor: Output tensor (size_batch, 1)
        '''
        x = self.backbone(x)
        x = self.head(x)

        return x
