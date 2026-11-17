# SPDX-License-Identifier: Apache-2.0
"""Gray-code encoding of per-token audio durations."""

import torch


def encode_time_as_gray_bits(frame_counts: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Encode integer frame counts as Gray-code bits in {-1, 1}, shape (..., num_bits)."""
    num_time_classes = 2**num_bits
    frame_counts = frame_counts.clamp(min=0, max=num_time_classes - 1)
    gray_code = frame_counts ^ (frame_counts >> 1)
    bit_positions = torch.arange(num_bits, device=frame_counts.device)
    gray_bits = (gray_code.unsqueeze(-1) >> bit_positions) & 1
    return (gray_bits.flip(-1).float() * 2.0) - 1.0


def decode_gray_bits_to_time(gray_bits: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Inverse of encode_time_as_gray_bits; returns integer frame counts."""
    bits = ((gray_bits + 1.0) / 2.0).round().long()
    # Bit 0 is the most significant bit, matching encode_time_as_gray_bits.
    bit_positions = torch.arange(num_bits - 1, -1, -1, device=gray_bits.device)
    gray_code = (bits * (1 << bit_positions)).sum(dim=-1)
    binary = gray_code
    shift = 1
    while shift < num_bits:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


__all__ = ["decode_gray_bits_to_time", "encode_time_as_gray_bits"]
