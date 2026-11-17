# SPDX-License-Identifier: Apache-2.0
"""Map TADA pipeline state onto SGLang's request lifecycle."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.tada.payload_types import (
    TadaState,
    load_tada_state,
    store_tada_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

TEXT_TEMPERATURE = 0.6
TEXT_TOP_P = 0.9
TEXT_REPETITION_PENALTY = 1.1


@dataclass
class TadaRequestData(SGLangARRequestData):
    state: TadaState = field(default_factory=TadaState)
    schedule: torch.Tensor | None = None
    prompt_acoustic: torch.Tensor | None = None
    prompt_acoustic_mask: torch.Tensor | None = None
    prompt_time_before: torch.Tensor | None = None
    prompt_time_after: torch.Tensor | None = None
    prefill_end: int = 0
    current_position: int = 0
    extra_steps: int = 0
    collected_frames: list[torch.Tensor] = field(default_factory=list)
    collected_time_before: list[torch.Tensor] = field(default_factory=list)
    feedback_acoustic: torch.Tensor | None = None
    feedback_mask: torch.Tensor | None = None
    feedback_time_before: torch.Tensor | None = None
    feedback_time_after: torch.Tensor | None = None
    prefill_hidden: torch.Tensor | None = None
    noise_generator: torch.Generator | None = None
    trailing_time_added: bool = False
    engine_start_s: float = 0.0
    stream: bool = False
    latest_stream_frame: torch.Tensor | None = None
    stream_chunk_id: int = 0


def build_tada_request(payload: StagePayload) -> TadaRequestData:
    state = load_tada_state(payload)
    if not state.input_ids:
        raise RuntimeError("TADA preprocessing did not build the text schedule")
    else:
        pass
    if state.prompt_acoustic is None or state.prompt_time_before is None:
        raise RuntimeError("TADA reference encoding did not produce prompt tensors")
    else:
        pass
    if state.prompt_acoustic_mask is None or state.prompt_time_after is None:
        raise RuntimeError("TADA prompt tensors are incomplete")
    else:
        pass
    schedule = torch.tensor(state.input_ids, dtype=torch.long).unsqueeze(0)
    prefill_end = int(state.prefill_len)
    if not 0 < prefill_end < schedule.shape[1]:
        raise ValueError(
            f"TADA prefill boundary {prefill_end} must fall inside the schedule "
            f"of {schedule.shape[1]} tokens"
        )
    else:
        pass
    budget = (schedule.shape[1] - prefill_end) + int(state.num_extra_steps)
    sampling_params = SamplingParams(
        max_new_tokens=budget,
        temperature=TEXT_TEMPERATURE,
        top_p=TEXT_TOP_P,
        repetition_penalty=TEXT_REPETITION_PENALTY,
        ignore_eos=True,
        stop_token_ids=[],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(state.vocab_size))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=schedule[0, :prefill_end].tolist(),
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=int(state.vocab_size),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    suppress_tokens = (
        [int(state.pad_token_id)] if state.pad_token_id is not None else None
    )
    return TadaRequestData(
        state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=schedule[:, :prefill_end],
        schedule=schedule,
        prompt_acoustic=state.prompt_acoustic.unsqueeze(0),
        prompt_acoustic_mask=state.prompt_acoustic_mask.unsqueeze(0),
        prompt_time_before=state.prompt_time_before.unsqueeze(0),
        prompt_time_after=state.prompt_time_after.unsqueeze(0),
        prefill_end=prefill_end,
        current_position=prefill_end,
        extra_steps=int(state.num_extra_steps),
        suppress_tokens=suppress_tokens,
        input_embeds_are_projected=True,
        engine_start_s=time.perf_counter(),
        stream=bool(payload.request.params.get("stream", False)),
    )


def apply_tada_result(data: TadaRequestData) -> StagePayload:
    state = data.state
    if not data.collected_frames:
        raise RuntimeError("TADA generated no acoustic frames")
    else:
        pass
    frames = torch.stack(data.collected_frames, dim=0)
    times = torch.stack(data.collected_time_before, dim=0)
    output_start = int(state.output_start)
    frames = frames[output_start:]
    times = times[output_start:]
    state.frames = frames * float(state.acoustic_std) + float(state.acoustic_mean)
    state.frame_time_before = times.reshape(-1)
    state.prompt_tokens = int(data.prefill_end)
    state.completion_tokens = int(frames.shape[0])
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.finish_reason = str(data.finish_reason or "")
    return store_tada_state(data.stage_payload, state)


def build_stream_output(
    request_id: str,
    data: TadaRequestData,
    req_output: RequestOutput,
) -> Iterator[OutgoingMessage]:
    """Emit one denormalised acoustic frame plus its duration per decode step."""
    del req_output
    frame = data.latest_stream_frame
    data.latest_stream_frame = None
    if frame is None or not data.stream:
        return
    else:
        pass
    metadata = {
        "modality": "audio_latents",
        "stream": True,
        "chunk_id": data.stream_chunk_id,
    }
    data.stream_chunk_id += 1
    yield OutgoingMessage(
        request_id=request_id,
        type="stream",
        target="vocoder",
        data=frame,
        metadata=metadata,
    )


__all__ = [
    "TadaRequestData",
    "apply_tada_result",
    "build_stream_output",
    "build_tada_request",
]
