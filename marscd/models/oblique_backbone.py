"""Module B - Streaming oblique encoder (online, T2 and T1 video).

A light ConvNeXt-style backbone + FPN producing stride-8/16 maps. In the
deployment configuration this is replaced by a distilled foundation backbone
(e.g. DINOv3 ConvNeXt-T / ViT-S/16 from the LVD-1689M release) through
`build_backbone`; the rest of the pipeline only relies on the FPN contract
(list of [1, C, h_l, w_l] maps + strides).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MARSConfig
from .common import ConvBNAct, ResBlock


class ObliqueBackbone(nn.Module):
    strides = (8, 16)

    def __init__(self, cfg: MARSConfig):
        super().__init__()
        c1, c2, c3 = cfg.img_channels
        C = cfg.d_model
        self.stem = nn.Sequential(ConvBNAct(3, c1 // 2, 3, 2), ConvBNAct(c1 // 2, c1, 3, 2))  # /4
        self.s8 = nn.Sequential(ConvBNAct(c1, c1, 3, 2), ResBlock(c1))
        self.s16 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), ResBlock(c2))
        self.s32 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), ResBlock(c3))
        self.lat = nn.ModuleList([nn.Conv2d(c, C, 1) for c in (c1, c2, c3)])
        self.smooth = nn.ModuleList([ConvBNAct(C, C, 3) for _ in range(2)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, img: Tensor) -> list[Tensor]:
        """img: [3, H, W] in [0, 1] -> [P8, P16] each [1, C, h, w]."""
        x = (img.unsqueeze(0) - self.mean) / self.std
        x = self.stem(x)
        f8 = self.s8(x)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        p32 = self.lat[2](f32)
        p16 = self.lat[1](f16) + F.interpolate(p32, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.lat[0](f8) + F.interpolate(p16, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        return [self.smooth[0](p8), self.smooth[1](p16)]


def build_backbone(cfg: MARSConfig, name: str = "light") -> nn.Module:
    if name == "light":
        return ObliqueBackbone(cfg)
    raise NotImplementedError(
        f"backbone '{name}': plug a pretrained encoder here (e.g. DINOv3 ConvNeXt-T via timm/transformers) "
        "and wrap it with an FPN that returns stride-8/16 maps of width cfg.d_model")
