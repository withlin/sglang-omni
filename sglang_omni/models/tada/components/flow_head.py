# SPDX-License-Identifier: Apache-2.0
"""VibeVoice flow-matching head and Euler solver for TADA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from transformers.activations import ACT2FN


def modulate(
    hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        float_states = hidden_states.float()
        normalized = float_states * torch.rsqrt(
            float_states.pow(2).mean(-1, keepdim=True) + self.eps
        )
        normalized = normalized.type_as(hidden_states)
        if self.weight is not None:
            normalized = normalized * self.weight
        else:
            pass
        return normalized


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            ACT2FN["silu"],
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        arguments = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        else:
            pass
        return embedding.to(timesteps.dtype)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        frequencies = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        frequencies = frequencies.to(self.mlp[0].weight.dtype)
        return self.mlp(frequencies)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.activation = ACT2FN["silu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class HeadLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        cond_dim: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            ACT2FN["silu"],
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def forward(
        self, hidden_states: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(conditioning).chunk(3, dim=-1)
        return hidden_states + gate * self.ffn(
            modulate(self.norm(hidden_states), shift, scale)
        )


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        cond_size: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            ACT2FN["silu"],
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def forward(
        self, hidden_states: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(hidden_states), shift, scale))


class TadaVibeVoiceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_layers: int,
        head_ffn_ratio: float,
        latent_size: int,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cond_dim = hidden_size
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, self.cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.cond_dim)
        ffn_dim = int(hidden_size * head_ffn_ratio)
        self.layers = nn.ModuleList(
            [
                HeadLayer(
                    embed_dim=hidden_size,
                    ffn_dim=ffn_dim,
                    cond_dim=self.cond_dim,
                    norm_eps=rms_norm_eps,
                )
                for _ in range(head_layers)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=rms_norm_eps,
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.noisy_images_proj(noisy_images)
        timestep_embedding = self.t_embedder(timesteps)
        conditioning = self.cond_proj(condition) + timestep_embedding
        for layer in self.layers:
            hidden_states = layer(hidden_states, conditioning)
        return self.final_layer(hidden_states, conditioning)


@dataclass(frozen=True)
class TadaFlowOptions:
    num_flow_matching_steps: int
    acoustic_cfg_scale: float
    duration_cfg_scale: float
    cfg_schedule: Literal["constant", "linear", "cosine"]
    time_schedule: Literal["uniform", "cosine", "logsnr"]


def build_time_schedule(
    num_steps: int, schedule: str, device: torch.device
) -> torch.Tensor:
    if schedule == "cosine":
        grid = torch.linspace(0, 1, num_steps + 1, device=device)
        return 0.5 * (1 - torch.cos(math.pi * grid))
    elif schedule == "logsnr":
        log_snr = torch.linspace(5.0, -5.0, num_steps + 1, device=device)
        span = torch.sigmoid(-log_snr / 2)
        span[0] = 0.0
        span[-1] = 1.0
        return span
    else:
        pass
    return torch.linspace(0, 1, num_steps + 1, device=device)


def scheduled_cfg(base_scale: float, timestep: float, schedule: str) -> float:
    if schedule == "constant" or base_scale == 1.0:
        return base_scale
    elif schedule == "linear":
        return 1.0 + (base_scale - 1.0) * (1.0 - timestep)
    elif schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * timestep))
    else:
        pass
    return base_scale


@torch.no_grad()
def solve_flow_matching(
    *,
    prediction_head: torch.nn.Module,
    speech: torch.Tensor,
    conditioning: torch.Tensor,
    negative_condition: torch.Tensor,
    acoustic_dim: int,
    options: TadaFlowOptions,
) -> torch.Tensor:
    time_span = build_time_schedule(
        options.num_flow_matching_steps, options.time_schedule, speech.device
    )
    for step_index in range(1, len(time_span)):
        delta = time_span[step_index] - time_span[step_index - 1]
        current = time_span[step_index - 1]
        acoustic_scale = scheduled_cfg(
            options.acoustic_cfg_scale, float(current), options.cfg_schedule
        )
        duration_scale = scheduled_cfg(
            options.duration_cfg_scale, float(current), options.cfg_schedule
        )
        if acoustic_scale != 1.0 or duration_scale != 1.0:
            doubled_speech = torch.cat([speech, speech], dim=0)
            doubled_condition = torch.cat(
                [conditioning, negative_condition], dim=0
            )
            timesteps = current.repeat(doubled_speech.shape[0]).to(doubled_speech)
            velocity = prediction_head(
                doubled_speech, timesteps, condition=doubled_condition
            )
            positive, negative = velocity.chunk(2, dim=0)
            guided = torch.cat(
                [
                    negative[:, :acoustic_dim]
                    + acoustic_scale
                    * (positive[:, :acoustic_dim] - negative[:, :acoustic_dim]),
                    negative[:, acoustic_dim:]
                    + duration_scale
                    * (positive[:, acoustic_dim:] - negative[:, acoustic_dim:]),
                ],
                dim=-1,
            )
        else:
            guided = prediction_head(
                speech,
                current.repeat(speech.shape[0]).to(speech),
                condition=conditioning,
            )
        speech = speech + delta * guided
    return speech


__all__ = [
    "TadaFlowOptions",
    "TadaVibeVoiceHead",
    "TimestepEmbedder",
    "build_time_schedule",
    "scheduled_cfg",
    "solve_flow_matching",
]
