# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from sglang_omni.models.tada.payload_types import TadaState
from sglang_omni.models.tada.request_builders import (
    TadaRequestData,
    build_stream_output,
)
from sglang_omni.models.tada.streaming_vocoder import (
    LATENT_DIM,
    PACKED_FRAME_WIDTH,
    TadaStreamState,
    TadaStreamingVocoder,
    TadaWindowPlan,
    build_window_plan,
)
from sglang_omni.proto import OmniRequest, StagePayload


class FrameIndexDecoder(nn.Module):
    """Map each latent row to its first channel repeated over the upsample factor."""

    def __init__(self, upsample: int = 480) -> None:
        super().__init__()
        self.upsample = upsample
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self, expanded: torch.Tensor, token_masks: torch.Tensor
    ) -> torch.Tensor:
        del token_masks
        indices = expanded[..., 0]
        waveform = indices.repeat_interleave(self.upsample, dim=1)
        return waveform.unsqueeze(1)


def make_vocoder() -> TadaStreamingVocoder:
    return TadaStreamingVocoder(
        FrameIndexDecoder(),
        chunk_frames=2,
        left_context_frames=1,
        right_context_frames=1,
    )


def make_chunk(frames: list[int], durations: list[int]) -> torch.Tensor:
    rows = torch.zeros(len(frames), PACKED_FRAME_WIDTH)
    for index, (frame, duration) in enumerate(zip(frames, durations)):
        rows[index, 0] = float(frame)
        rows[index, LATENT_DIM] = float(duration)
    return rows


def feed(
    vocoder: TadaStreamingVocoder,
    request_id: str,
    state: TadaStreamState,
    frames: list[int],
    durations: list[int],
) -> None:
    chunk = make_chunk(frames, durations)
    validated = vocoder.validate_chunk(request_id, state, chunk)
    vocoder.ingest(request_id, state, validated)


def frame_samples(waveform: torch.Tensor, upsample: int = 480) -> list[float]:
    values = waveform.reshape(-1, upsample)[:, 0].tolist()
    return [round(value) for value in values]


def test_streaming_windows_emit_only_new_frames() -> None:
    vocoder = make_vocoder()
    state = vocoder.create_stream_state("rid")
    emitted: list[float] = []
    for frame in (1, 2, 3, 4, 5):
        feed(vocoder, "rid", state, [frame], [1])
        if vocoder.should_decode(state, is_final=False):
            audio = vocoder.decode_delta("rid", state, is_final=False)
            assert audio is not None
            emitted.extend(frame_samples(audio))
    assert emitted == [1, 2, 3, 4]


def test_first_token_drops_leading_silence_and_later_gaps_expand() -> None:
    vocoder = make_vocoder()
    state = vocoder.create_stream_state("rid")
    feed(vocoder, "rid", state, [1], [3])
    assert len(state.expanded_frames) == 1
    feed(vocoder, "rid", state, [2], [3])
    assert len(state.expanded_frames) == 4
    assert [int(row[0].item()) for row in state.expanded_frames] == [1, 0, 0, 2]


def test_final_flush_appends_trailing_duration_and_remaining_frames() -> None:
    vocoder = make_vocoder()
    state = vocoder.create_stream_state("rid")
    for frame in (1, 2, 3, 4, 5, 6):
        feed(vocoder, "rid", state, [frame], [1])
        if vocoder.should_decode(state, is_final=False):
            vocoder.decode_delta("rid", state, is_final=False)
    vocoder.stream_payloads["rid"] = StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "hello"}, params={}),
        data=TadaState(
            frame_time_before=torch.tensor([1, 1, 1, 1, 1, 1, 2]),
        ).to_dict(),
    )
    audio = vocoder.decode_delta("rid", state, is_final=True)
    assert audio is not None
    assert frame_samples(audio) == [5, 6, 0, 0]


def test_window_plan_is_pure_and_returns_none_before_threshold() -> None:
    plan = build_window_plan(
        total_frames=10,
        emitted_frames=4,
        chunk_frames=2,
        left_context_frames=1,
        right_context_frames=1,
        is_final=False,
    )
    assert plan == TadaWindowPlan(
        window_start_frame=3,
        window_end_frame=7,
        emit_frames=2,
        emit_sample_offset=1,
    )
    assert (
        build_window_plan(
            total_frames=2,
            emitted_frames=0,
            chunk_frames=2,
            left_context_frames=1,
            right_context_frames=1,
            is_final=False,
        )
        is None
    )
    final = build_window_plan(
        total_frames=2,
        emitted_frames=0,
        chunk_frames=2,
        left_context_frames=1,
        right_context_frames=1,
        is_final=True,
    )
    assert final == TadaWindowPlan(
        window_start_frame=0,
        window_end_frame=2,
        emit_frames=2,
        emit_sample_offset=0,
    )


def test_coalesced_step_batches_streams_by_window_length() -> None:
    vocoder = make_vocoder()
    state_a = vocoder.create_stream_state("a")
    state_b = vocoder.create_stream_state("b")
    vocoder.stream_states["a"] = state_a
    vocoder.stream_states["b"] = state_b
    for frame in (1, 2, 3):
        feed(vocoder, "a", state_a, [frame], [1])
    audio = vocoder.decode_delta("a", state_a, is_final=False)
    assert audio is not None
    assert frame_samples(audio) == [1, 2]
    for frame in (4, 5):
        feed(vocoder, "a", state_a, [frame], [1])
    for frame in (1, 2, 3):
        feed(vocoder, "b", state_b, [frame], [1])
    participants = vocoder.select_step_participants()
    assert [request_id for request_id, _ in participants] == ["a", "b"]
    plans = vocoder.build_step_plan(participants)
    assert [
        plan.window_end_frame - plan.window_start_frame for plan in plans
    ] == [4, 3]
    audios = vocoder.run_step(participants, plans)
    assert frame_samples(audios["a"]) == [3, 4]
    assert frame_samples(audios["b"]) == [1, 2]


def test_select_step_participants_respects_max_batch_size() -> None:
    vocoder = TadaStreamingVocoder(
        FrameIndexDecoder(),
        chunk_frames=2,
        left_context_frames=1,
        right_context_frames=1,
        max_batch_size=1,
    )
    for request_id in ("a", "b"):
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        for frame in (1, 2, 3):
            feed(vocoder, request_id, state, [frame], [1])
    participants = vocoder.select_step_participants()
    assert len(participants) == 1


def test_non_streaming_decode_trims_the_first_duration() -> None:
    vocoder = TadaStreamingVocoder(FrameIndexDecoder(), chunk_frames=2)
    frames = torch.zeros(3, LATENT_DIM)
    for index in range(3):
        frames[index, 0] = float(index + 1)
    times = torch.ones(4, dtype=torch.long)
    waveform = vocoder.decode_frames(frames, times)
    assert frame_samples(waveform) == [2, 3, 0]


def test_stream_chunk_validation_rejects_wrong_width() -> None:
    vocoder = make_vocoder()
    state = vocoder.create_stream_state("rid")
    try:
        vocoder.validate_chunk("rid", state, torch.zeros(1, LATENT_DIM))
    except ValueError as error:
        assert "must be [frames, 513]" in str(error)
    else:
        raise AssertionError("expected a width validation error")


def test_build_stream_output_emits_then_clears_the_frame() -> None:
    data = TadaRequestData()
    data.stream = True
    data.latest_stream_frame = torch.ones(1, PACKED_FRAME_WIDTH)
    messages = list(build_stream_output("rid", data, None))
    assert len(messages) == 1
    assert messages[0].target == "vocoder"
    assert messages[0].metadata == {
        "modality": "audio_latents",
        "stream": True,
        "chunk_id": 0,
    }
    assert data.stream_chunk_id == 1
    assert data.latest_stream_frame is None
    assert list(build_stream_output("rid", data, None)) == []
