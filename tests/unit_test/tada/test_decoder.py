# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.tada.components.codec_decoder import expand_frames


def test_expand_frames_inserts_silence_before_each_frame_and_tail() -> None:
    frames = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    time_before = torch.tensor([[1, 3, 2, 1]])
    expanded, frame_mask = expand_frames(frames, time_before)
    assert expanded.shape == (1, 7, 4)
    assert frame_mask.tolist() == [[1, 0, 0, 1, 0, 1, 0]]
    assert torch.equal(expanded[0, 0], frames[0, 0])
    assert torch.equal(expanded[0, 3], frames[0, 1])
    assert torch.equal(expanded[0, 5], frames[0, 2])
