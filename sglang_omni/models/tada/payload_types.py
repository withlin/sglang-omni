# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried by the TADA pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class TadaState(DeclarativeStateBase):
    input_text: str = wire("", codec="str")
    reference_text: str = wire("", codec="str")
    reference_audio: str | None = wire(None, codec="str_or")
    language: str = wire("en", codec="str")
    input_ids: list[int] = wire(default_factory=list, codec="list")
    prompt_acoustic: torch.Tensor | None = wire(None, codec="typed_tensor")
    prompt_acoustic_mask: torch.Tensor | None = wire(None, codec="typed_tensor")
    prompt_time_before: torch.Tensor | None = wire(None, codec="typed_tensor")
    prompt_time_after: torch.Tensor | None = wire(None, codec="typed_tensor")
    prefill_len: int = wire(0, codec="int")
    output_start: int = wire(0, codec="int")
    num_extra_steps: int = wire(0, codec="int")
    seed: int | None = wire(None, codec="opt_int")
    acoustic_mean: float = wire(0.0, codec="float")
    acoustic_std: float = wire(1.5, codec="float")
    vocab_size: int = wire(128256, codec="int")
    pad_token_id: int | None = wire(None, codec="opt_int")
    frames: torch.Tensor | None = wire(None, codec="typed_tensor")
    frame_time_before: torch.Tensor | None = wire(None, codec="typed_tensor")


def load_tada_state(payload: StagePayload) -> TadaState:
    return TadaState.from_dict(payload.data)


def store_tada_state(payload: StagePayload, state: TadaState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


__all__ = ["TadaState", "load_tada_state", "store_tada_state"]
