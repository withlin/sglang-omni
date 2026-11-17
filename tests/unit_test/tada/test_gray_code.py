# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.tada.components.gray_code import (
    decode_gray_bits_to_time,
    encode_time_as_gray_bits,
)


def test_gray_code_round_trips_every_time_class() -> None:
    num_bits = 8
    frames = torch.arange(2**num_bits)
    bits = encode_time_as_gray_bits(frames, num_bits)
    assert bits.shape == (2**num_bits, num_bits)
    assert set(bits.unique().tolist()) <= {-1.0, 1.0}
    assert torch.equal(decode_gray_bits_to_time(bits, num_bits), frames)


def test_gray_code_clamps_out_of_range_frames() -> None:
    bits = encode_time_as_gray_bits(torch.tensor([-5, 300]), 8)
    assert decode_gray_bits_to_time(bits, 8).tolist() == [0, 255]


def test_gray_code_bit_order_matches_the_checkpoint() -> None:
    bits = encode_time_as_gray_bits(torch.tensor([1]), 8)
    assert bits.tolist() == [[-1.0] * 7 + [1.0]]
    assert decode_gray_bits_to_time(bits, 8).tolist() == [1]
