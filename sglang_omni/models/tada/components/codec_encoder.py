# SPDX-License-Identifier: Apache-2.0
"""Reference-audio encoder for TADA prompts."""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.tada.components.codec_attention import (
    LocalAttentionEncoder,
    create_encoder_attention_mask,
)
from sglang_omni.models.tada.components.codec_layers import WavEncoder


@dataclass
class TadaEncoderOutput:
    token_values: torch.Tensor
    token_positions: torch.Tensor
    token_masks: torch.Tensor
    audio_length: torch.Tensor


class TadaEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        embed_dim: int = 512,
        strides: tuple[int, ...] = (6, 5, 4, 4),
        num_attn_layers: int = 6,
        num_attn_heads: int = 8,
        attn_dim_feedforward: int = 4096,
        attn_dropout: float = 0.1,
        block_attention: Literal["none", "v1", "v2"] = "v2",
        max_seq_len: int = 8192,
        std: float = 0.5,
        acoustic_mean: float = 0.0,
        acoustic_std: float = 1.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.block_attention = block_attention
        self.std = std
        self.acoustic_mean = acoustic_mean
        self.acoustic_std = acoustic_std
        self.wav_encoder = WavEncoder(
            d_model=64, strides=strides, d_latent=hidden_dim
        )
        self.local_attention_encoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
            dropout=attn_dropout,
            activation="gelu",
            max_seq_len=max_seq_len,
        )
        self.hidden_linear = (
            nn.Linear(hidden_dim, embed_dim)
            if hidden_dim != embed_dim
            else nn.Identity()
        )
        self.pos_emb = nn.Embedding(2, hidden_dim)

    def get_encoder_outputs(
        self, audio: torch.Tensor, token_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.wav_encoder(F.pad(audio.unsqueeze(1), (0, 960))).transpose(
            1, 2
        )
        sequence_length = encoded.shape[1]
        if token_masks.shape[1] > sequence_length:
            padded_token_masks = token_masks[:, :sequence_length]
        else:
            padded_token_masks = F.pad(
                token_masks,
                (0, sequence_length - token_masks.shape[1]),
                value=0,
            )
        encoded = encoded + self.pos_emb(padded_token_masks)
        attention_mask = create_encoder_attention_mask(
            padded_token_masks, version=self.block_attention
        )
        encoded = self.local_attention_encoder(encoded, mask=attention_mask)
        encoded = self.hidden_linear(encoded)
        return encoded, padded_token_masks

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        text_token_positions: torch.Tensor,
        token_masks: torch.Tensor,
        sample: bool = True,
        noise_std: float | None = None,
    ) -> TadaEncoderOutput:
        audio = audio.to(next(self.parameters()).dtype)
        audio_length = torch.tensor(
            [audio.shape[-1]], device=audio.device, dtype=torch.long
        )
        encoded, padded_token_masks = self.get_encoder_outputs(audio, token_masks)
        encoded = torch.where(
            padded_token_masks.unsqueeze(-1) == 0, torch.zeros_like(encoded), encoded
        )
        std = self.resolved_noise_std(noise_std)
        if sample and std > 0:
            noise = torch.randn_like(encoded) * std
            encoded = torch.where(
                padded_token_masks.unsqueeze(-1) == 0,
                encoded,
                encoded + noise,
            )
        else:
            pass
        gather_index = (text_token_positions - 1).clamp(min=0).unsqueeze(-1)
        gather_index = gather_index.expand(
            -1, -1, encoded.shape[-1]
        )
        token_values = torch.gather(encoded, 1, gather_index)
        token_values = (token_values - self.acoustic_mean) / self.acoustic_std
        return TadaEncoderOutput(
            token_values=token_values,
            token_positions=text_token_positions,
            token_masks=padded_token_masks,
            audio_length=audio_length,
        )

    def resolved_noise_std(self, noise_std: float | None) -> float:
        if noise_std is None:
            return self.std
        else:
            pass
        return float(noise_std)


__all__ = ["TadaEncoder", "TadaEncoderOutput"]
