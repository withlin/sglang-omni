# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.models.tada.sglang_model import classify_tada_weight


def test_classify_routes_llama_head_and_skips_decoder() -> None:
    assert classify_tada_weight("model.embed_tokens.weight") == "llama"
    assert classify_tada_weight("model.layers.0.self_attn.q_proj.weight") == "llama"
    assert classify_tada_weight("acoustic_proj.weight") == "head"
    assert classify_tada_weight("time_start_embed.weight") == "head"
    assert classify_tada_weight("bottleneck_proj.weight") == "head"
    assert classify_tada_weight("prediction_head.cond_proj.weight") == "head"
    assert classify_tada_weight("_decoder.decoder_proj.weight") == "skip"
    with pytest.raises(ValueError, match="Unexpected TADA checkpoint weight"):
        classify_tada_weight("encoder.wav2vec2.weight")
