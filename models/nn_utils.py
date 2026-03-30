"""Neural network helpers — GroupNorm32, normalization, zero_module, timestep_embedding."""

import math
import torch
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    """GroupNorm computed in float32 to avoid numerical instability under mixed precision."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    """Standard normalization: 32-group GroupNorm (matches DIFUSCO)."""
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-init all params; residual branches start at zero."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding. [cos, sin] order (matches DIFUSCO)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
