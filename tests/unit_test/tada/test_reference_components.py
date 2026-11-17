# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.tada.components.codec_aligner import (
    check_alignment,
    wav2vec2_large_config,
)
from sglang_omni.models.tada.components.codec_attention import (
    create_decoder_attention_mask,
    create_encoder_attention_mask,
)


def test_wav2vec2_large_config_matches_the_aligner_checkpoint() -> None:
    config = wav2vec2_large_config(vocab_size=128256)
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 16
    assert config.conv_dim == [512] * 7
    assert config.vocab_size == 128256


def test_encoder_and_decoder_masks_differ_at_block_starts() -> None:
    token_mask = torch.tensor([[1, 0, 0, 1, 0]])
    encoder_mask = create_encoder_attention_mask(token_mask, version="v2")
    decoder_mask = create_decoder_attention_mask(token_mask, version="v2")
    assert encoder_mask.shape == (1, 5, 5)
    assert decoder_mask.shape == (1, 5, 5)
    # The encoder hides block-start markers from unmarked positions.
    assert encoder_mask[0, 1, 0]
    assert not decoder_mask[0, 1, 0]
    # A marked position reads the previous block but not its marker.
    assert encoder_mask[0, 3, 0]
    assert not encoder_mask[0, 3, 1]
    assert not decoder_mask[0, 3, 0]
    assert encoder_mask[0, 0, 4]
    assert not encoder_mask[0, 1, 2]


def test_alignment_report_flags_bunching_long_gaps_and_order() -> None:
    report = check_alignment(
        token_positions=torch.tensor([[1, 2, 3, 4, 5, 200]]),
        audio_frames=300,
    )
    assert report.is_monotonic
    assert report.has_bunching
    assert report.has_long_gap
    broken = check_alignment(
        token_positions=torch.tensor([[1, 5, 4, 400]]),
        audio_frames=300,
    )
    assert not broken.is_monotonic
    assert broken.is_out_of_range
