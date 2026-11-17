# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.tada.payload_types import TadaState
from sglang_omni.models.tada.request_builders import (
    apply_tada_result,
    build_tada_request,
)
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(state: TadaState) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "hello"}, params={}),
        data=state.to_dict(),
    )


def make_state() -> TadaState:
    frames = torch.zeros(7, 4)
    times = torch.ones(7, dtype=torch.long)
    return TadaState(
        input_ids=[
            1,
            5,
            6,
            7,
            8,
            9,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            200,
            201,
            3,
            3,
            3,
            3,
            3,
        ],
        prompt_acoustic=frames,
        prompt_acoustic_mask=torch.ones(7, dtype=torch.long),
        prompt_time_before=times,
        prompt_time_after=times,
        prefill_len=11,
        output_start=11,
        num_extra_steps=0,
        pad_token_id=2,
    )


def test_budget_is_remaining_positions_plus_extra_steps() -> None:
    state = make_state()
    data = build_tada_request(make_payload(state))
    assert data.req.origin_input_ids == state.input_ids[:11]
    assert data.req.sampling_params.max_new_tokens == 9
    assert data.req.sampling_params.temperature == 0.6
    assert data.req.sampling_params.top_p == 0.9
    assert data.req.sampling_params.repetition_penalty == 1.1
    assert data.req.sampling_params.ignore_eos is True
    assert data.suppress_tokens == [2]


def test_result_trims_frames_and_denormalizes(monkeypatch) -> None:
    state = make_state()
    data = build_tada_request(make_payload(state))
    data.collected_frames = [torch.ones(4) for _ in range(15)]
    data.collected_time_before = [
        torch.ones(1, dtype=torch.long) for _ in range(16)
    ]
    data.engine_start_s = 10.0
    monkeypatch.setattr(
        "sglang_omni.models.tada.request_builders.time.perf_counter",
        lambda: 12.0,
    )
    payload = apply_tada_result(data)
    restored = TadaState.from_dict(payload.data)
    assert restored.frames.shape == (4, 4)
    assert torch.allclose(restored.frames, torch.full((4, 4), 1.5))
    assert restored.frame_time_before.shape == (5,)
    assert restored.engine_time_s == 2.0
