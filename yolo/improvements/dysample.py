"""DySample dynamic upsampling module.

This is a lightweight implementation of the "lp" DySample variant. It replaces
nearest-neighbor upsampling in the YOLO neck with learned sampling offsets while
preserving the input channel count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Dynamic upsampling with learned offsets.

    Args:
        channels: Number of input and output channels.
        scale: Spatial upsampling factor.
        groups: Channel groups used by grid_sample.
        dyscope: If true, learn a scope gate for the offsets.
    """

    def __init__(self, channels: int, scale: int = 2, groups: int = 4, dyscope: bool = False):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
        self.channels = channels
        self.scale = scale
        self.groups = groups
        self.dyscope = dyscope

        offset_channels = 2 * groups * scale * scale
        self.offset = nn.Conv2d(channels, offset_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(channels, offset_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._make_init_pos())

    def _make_init_pos(self) -> torch.Tensor:
        coord = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        pos = torch.stack([xx, yy], dim=0)
        return pos.transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        y = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        x_coord = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).view(1, 2, 1, h, w)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0

        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        x = x.reshape(b * self.groups, -1, h, w)
        sampled = F.grid_sample(x, coords, mode="bilinear", align_corners=False, padding_mode="border")
        return sampled.view(b, -1, self.scale * h, self.scale * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dyscope:
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)
