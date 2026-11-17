# SPDX-License-Identifier: Apache-2.0
"""DAC decoder and frame expansion for TADA waveforms."""

import math

import torch
from torch import nn

from sglang_omni.models.tada.components.codec_attention import (
    LocalAttentionEncoder,
    create_decoder_attention_mask,
)
from sglang_omni.models.tada.components.codec_layers import (
    ResidualUnit,
    Snake1d,
    WNConv1d,
    WNConvTranspose1d,
)


class DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int = 16, output_dim: int = 8, stride: int = 1
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class DACDecoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: tuple[int, ...],
        d_out: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            WNConv1d(input_channel, channels, kernel_size=7, padding=3)
        ]
        output_dim = channels
        for index, stride in enumerate(rates):
            input_dim = channels // 2**index
            output_dim = channels // 2 ** (index + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride))
        layers.extend(
            [
                Snake1d(output_dim),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


class TadaDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_attn_layers: int = 6,
        num_attn_heads: int = 8,
        attn_dim_feedforward: int = 4096,
        wav_decoder_channels: int = 1536,
        strides: tuple[int, ...] = (4, 4, 5, 6),
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        self.decoder_proj = nn.Linear(embed_dim, hidden_dim)
        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
            dropout=0.1,
            activation="gelu",
            max_seq_len=max_seq_len,
        )
        self.wav_decoder = DACDecoder(
            input_channel=hidden_dim,
            channels=wav_decoder_channels,
            rates=strides,
        )
        self.upsample_factor = math.prod(strides)

    def forward(
        self, encoded_expanded: torch.Tensor, token_masks: torch.Tensor
    ) -> torch.Tensor:
        decoder_input = self.decoder_proj(encoded_expanded)
        attention_mask = create_decoder_attention_mask(token_masks, version="v2")
        decoded = self.local_attention_decoder(decoder_input, mask=attention_mask)
        return self.wav_decoder(decoded.transpose(1, 2))


def expand_frames(
    frames: torch.Tensor, time_before: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert silence frames before each token and pad a batch to the longest row."""
    batch_size, frame_count, feature_dim = frames.shape
    if time_before.shape[1] < frame_count + 1:
        raise ValueError(
            "TADA time_before must hold one value per frame plus a trailing value; "
            f"got {time_before.shape[1]} values for {frame_count} frames"
        )
    else:
        pass
    expanded_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for row in range(batch_size):
        parts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for frame_index in range(frame_count):
            zero_count = int(max(int(time_before[row, frame_index].item()) - 1, 0))
            if zero_count > 0:
                parts.append(
                    torch.zeros(
                        zero_count,
                        feature_dim,
                        device=frames.device,
                        dtype=frames.dtype,
                    )
                )
                masks.append(
                    torch.zeros(zero_count, dtype=torch.long, device=frames.device)
                )
            else:
                pass
            parts.append(frames[row, frame_index].unsqueeze(0))
            masks.append(
                torch.ones(1, dtype=torch.long, device=frames.device)
            )
        tail_count = int(max(int(time_before[row, frame_count].item()), 0))
        if tail_count > 0:
            parts.append(
                torch.zeros(
                    tail_count,
                    feature_dim,
                    device=frames.device,
                    dtype=frames.dtype,
                )
            )
            masks.append(
                torch.zeros(tail_count, dtype=torch.long, device=frames.device)
            )
        else:
            pass
        expanded_rows.append(torch.cat(parts, dim=0))
        mask_rows.append(torch.cat(masks, dim=0))
    longest = max(row.shape[0] for row in expanded_rows)
    padded = torch.zeros(
        batch_size, longest, feature_dim, device=frames.device, dtype=frames.dtype
    )
    padded_mask = torch.zeros(
        batch_size, longest, dtype=torch.long, device=frames.device
    )
    for row in range(batch_size):
        length = expanded_rows[row].shape[0]
        padded[row, :length] = expanded_rows[row]
        padded_mask[row, :length] = mask_rows[row]
    return padded, padded_mask


__all__ = ["DACDecoder", "DecoderBlock", "TadaDecoder", "expand_frames"]
