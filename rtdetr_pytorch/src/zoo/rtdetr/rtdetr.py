"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from src.core import register

__all__ = ['RTDETR', ]


# 主模型,很少的代码
@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        # 图像的多种尺寸 [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        # 随机选择一种图像尺寸，对x（输入的图像）进行插值，进行缩放
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        # 经过backbone
        x = self.backbone(x)
        # 经过encoder, HybridEncoder
        x = self.encoder(x)
        # 经过decoder, RTDETRTransformer (这里其实只有正常意义上的decoder)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
