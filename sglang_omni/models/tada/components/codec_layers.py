# SPDX-License-Identifier: Apache-2.0
"""Shared codec layers for the TADA encoder and decoder."""

import math

import torch
from torch import nn


class Snake1d(nn.Module):
    """Snake activation kept under the checkpoint parameter name alpha."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flattened = hidden_states.reshape(shape[0], shape[1], -1)
        activated = flattened + (self.alpha + 1e-9).reciprocal() * torch.sin(
            self.alpha * flattened
        ).pow(2)
        return activated.reshape(shape)


def WNConv1d(*args: object, **kwargs: object) -> nn.Conv1d:
    return torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args: object, **kwargs: object) -> nn.ConvTranspose1d:
    return torch.nn.utils.parametrizations.weight_norm(
        nn.ConvTranspose1d(*args, **kwargs)
    )


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1) -> None:
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=padding),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        padding = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if padding > 0:
            hidden_states = hidden_states[..., padding:-padding]
        else:
            pass
        return hidden_states + residual


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class WavEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: tuple[int, ...] = (2, 4, 8, 8),
        d_latent: int = 64,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(EncoderBlock(d_model, stride=stride))
        blocks.extend(
            [
                Snake1d(d_model),
                WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
            ]
        )
        self.block = nn.Sequential(*blocks)
        self.encoded_dim = d_model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.block(waveform)


__all__ = [
    "EncoderBlock",
    "ResidualUnit",
    "Snake1d",
    "WNConv1d",
    "WNConvTranspose1d",
    "WavEncoder",
]
