"""Shared neural building blocks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int = 2, act=nn.GELU):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d_hidden), act()]
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def sinusoid(x: Tensor, dim: int) -> Tensor:
    """Transformer-style sinusoidal encoding of a scalar tensor: [...] -> [..., dim]."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
    a = x.unsqueeze(-1) * freq
    return torch.cat([a.sin(), a.cos()], -1)


class ScalarEmbedding(nn.Module):
    """sinusoid(x / sigma) -> Linear; GeoTransformer-style distance/angle embedding."""

    def __init__(self, d_out: int, sigma: float, d_sin: int | None = None):
        super().__init__()
        self.sigma = sigma
        self.d_sin = d_sin or d_out
        self.proj = nn.Linear(self.d_sin, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(sinusoid(x / self.sigma, self.d_sin))


class FourierFeatures(nn.Module):
    """Log-spaced Fourier features of D-dim coordinates (metres) -> MLP."""

    def __init__(self, d_in: int, d_out: int, n_freq: int = 8, min_period: float = 1.0, max_period: float = 512.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods, persistent=False)
        self.mlp = MLP(d_in * n_freq * 2 + d_in, d_out, d_out)

    def forward(self, x: Tensor) -> Tensor:
        a = x.unsqueeze(-1) * self.omega  # [..., D, F]
        feats = torch.cat([a.sin(), a.cos()], -1).flatten(-2)
        return self.mlp(torch.cat([feats, x / 100.0], -1))


class RelationalAttention(nn.Module):
    """Multi-head attention with relation-aware keys and additive bias.

    logits_ij = q_i . (k_j + W_r r_ij) / sqrt(d_h) + b_ij      (GeoTransformer form)

    All inputs carry a leading batch dimension:
      x_q [B, Lq, C], x_kv [B, Lk, C], rel [B, Lq, Lk, R], bias [B, H, Lq, Lk],
      mask [B, Lq, Lk] (True = attend). Fully-masked rows output zeros.
    """

    def __init__(self, d: int, n_heads: int, rel_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.r = nn.Linear(rel_dim, d) if rel_dim else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q: Tensor, x_kv: Tensor, rel: Tensor | None = None, bias: Tensor | None = None,
                mask: Tensor | None = None) -> Tensor:
        B, Lq, C = x_q.shape
        Lk = x_kv.shape[1]
        q = self.q(x_q).view(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(x_kv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(x_kv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        logits = q @ k.transpose(-1, -2)
        if rel is not None and self.r is not None:
            r = self.r(rel).view(B, Lq, Lk, self.h, self.dh).permute(0, 3, 1, 2, 4)
            logits = logits + (q.unsqueeze(-2) * r).sum(-1)
        logits = logits / math.sqrt(self.dh)
        if bias is not None:
            logits = logits + bias
        if mask is not None:
            m = mask.unsqueeze(1)
            logits = logits.masked_fill(~m, float("-inf"))
            attn = torch.softmax(logits, -1).nan_to_num(0.0)
            attn = attn * m
        else:
            attn = torch.softmax(logits, -1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, Lq, C)
        return self.o(out)


class FFN(nn.Module):
    def __init__(self, d: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * mult, d))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvBNAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )


class ResBlock(nn.Module):
    """Depth-wise separable residual block (ConvNeXt-like, BN for streaming speed)."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.pw = nn.Sequential(nn.Conv2d(c, 4 * c, 1), nn.GELU(), nn.Conv2d(4 * c, c, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pw(self.bn(self.dw(x)))


def masked_mean(x: Tensor, w: Tensor, dim: int) -> Tensor:
    return (x * w).sum(dim) / w.sum(dim).clamp_min(1e-6)


def grid_sample_points(fmap: Tensor, uv: Tensor, hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Bilinearly sample a [C, h, w] map at pixel coords uv [..., 2] given in an image of size hw.

    Returns features [..., C] and an in-bounds mask [...].
    """
    H, W = hw
    gx = uv[..., 0] / W * 2 - 1
    gy = uv[..., 1] / H * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
    out = F.grid_sample(fmap.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out.reshape(fmap.shape[0], -1).t().reshape(*uv.shape[:-1], fmap.shape[0])
    valid = (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
    return out, valid
