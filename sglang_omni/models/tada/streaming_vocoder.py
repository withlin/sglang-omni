# SPDX-License-Identifier: Apache-2.0
"""Whole-utterance and windowed streaming decode for the TADA DAC codec."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from sglang_omni.models.tada.components.codec_decoder import TadaDecoder, expand_frames
from sglang_omni.models.tada.payload_types import load_tada_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 50
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAMES_PER_SECOND
LATENT_DIM = 512
PACKED_FRAME_WIDTH = LATENT_DIM + 1
DECODER_MAX_FRAMES = 8192
DEFAULT_CHUNK_FRAMES = 50
DEFAULT_LEFT_CONTEXT_FRAMES = 128
DEFAULT_RIGHT_CONTEXT_FRAMES = 32
DEFAULT_STREAM_MAX_BATCH_SIZE = 8


@dataclass
class TadaStreamState:
    expanded_frames: list[torch.Tensor] = field(default_factory=list)
    emitted_frames: int = 0
    started: bool = False
    trailing_resolved: bool = False
    trailing_frames: int = 0


@dataclass(frozen=True)
class TadaWindowPlan:
    window_start_frame: int
    window_end_frame: int
    emit_frames: int
    emit_sample_offset: int


def build_window_plan(
    *,
    total_frames: int,
    emitted_frames: int,
    chunk_frames: int,
    left_context_frames: int,
    right_context_frames: int,
    is_final: bool,
) -> TadaWindowPlan | None:
    """Pure window plan: which frames to decode and which to emit."""
    pending_frames = total_frames - emitted_frames
    if pending_frames <= 0:
        return None
    elif not is_final and pending_frames < chunk_frames + right_context_frames:
        return None
    else:
        pass
    emit_frames = pending_frames if is_final else chunk_frames
    window_right_frames = 0 if is_final else right_context_frames
    window_start_frame = max(0, emitted_frames - left_context_frames)
    window_end_frame = min(
        total_frames, emitted_frames + emit_frames + window_right_frames
    )
    return TadaWindowPlan(
        window_start_frame=window_start_frame,
        window_end_frame=window_end_frame,
        emit_frames=emit_frames,
        emit_sample_offset=emitted_frames - window_start_frame,
    )


class TadaStreamingVocoder(
    StreamingVocoderBase[TadaStreamState, list[TadaWindowPlan]]
):
    """Decode overlapping dense-frame windows of the non-causal TADA codec."""

    can_batch_stream_chunks = True

    def __init__(
        self,
        decoder: TadaDecoder,
        *,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        left_context_frames: int = DEFAULT_LEFT_CONTEXT_FRAMES,
        right_context_frames: int = DEFAULT_RIGHT_CONTEXT_FRAMES,
        max_batch_size: int = DEFAULT_STREAM_MAX_BATCH_SIZE,
    ) -> None:
        if chunk_frames < 1:
            raise ValueError("TADA stream_chunk_frames must be positive")
        else:
            pass
        if left_context_frames < 0 or right_context_frames < 0:
            raise ValueError("TADA stream context frames must be non-negative")
        else:
            pass
        if max_batch_size < 1:
            raise ValueError("TADA stream_max_batch_size must be positive")
        else:
            pass
        if (
            left_context_frames + chunk_frames + right_context_frames
            > DECODER_MAX_FRAMES
        ):
            raise ValueError(
                "TADA stream window must fit the decoder limit of "
                f"{DECODER_MAX_FRAMES} frames"
            )
        else:
            pass
        self.decoder = decoder
        self.chunk_frames = int(chunk_frames)
        self.left_context_frames = int(left_context_frames)
        self.right_context_frames = int(right_context_frames)
        self.max_batch_size = int(max_batch_size)
        self.stream_chunk_batch_max = int(max_batch_size)
        self.device = next(decoder.parameters()).device
        self.dtype = next(decoder.parameters()).dtype
        self.lock = threading.Lock()
        self.silence_frame = torch.zeros(LATENT_DIM, dtype=torch.float32)
        super().__init__(
            self.decode_full,
            sample_rate=SAMPLE_RATE,
            stream_source_hint="TADA",
            stream_input_modality="audio_latents",
        )

    def decode_full(self, payload: StagePayload) -> StagePayload:
        """Decode the whole utterance for a non-streaming request."""
        state = load_tada_state(payload)
        if state.frames is None or state.frame_time_before is None:
            raise RuntimeError("TADA vocoder received no acoustic frames")
        else:
            pass
        waveform = self.decode_frames(state.frames, state.frame_time_before)
        payload.data.update(
            audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint="TADA",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        else:
            pass
        return payload

    def decode_frames(
        self, frames: torch.Tensor, time_before: torch.Tensor
    ) -> torch.Tensor:
        """Expand durations, decode, and trim the leading silence."""
        frames = frames.to(self.device)
        time_before = time_before.to(self.device)
        expanded, token_masks = expand_frames(
            frames.unsqueeze(0), time_before.unsqueeze(0)
        )
        if expanded.shape[1] > DECODER_MAX_FRAMES:
            raise ValueError(
                "TADA expanded audio exceeds the decoder limit of "
                f"{DECODER_MAX_FRAMES} frames "
                f"(~{DECODER_MAX_FRAMES / FRAMES_PER_SECOND:.0f} s); "
                "stream the request or shorten the target text"
            )
        else:
            pass
        with self.lock, torch.inference_mode():
            waveform = self.decoder(
                expanded.to(dtype=self.dtype), token_masks=token_masks
            )
        waveform = waveform[0, 0].float().cpu()
        trim_samples = int(
            SAMPLE_RATE * int(time_before[0].item()) / FRAMES_PER_SECOND
        )
        return waveform[min(trim_samples, waveform.shape[0]) :]

    def create_stream_state(self, request_id: str) -> TadaStreamState:
        del request_id
        return TadaStreamState()

    def validate_chunk(
        self, request_id: str, state: TadaStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del state
        if codes.ndim != 2 or codes.shape[1] != PACKED_FRAME_WIDTH:
            raise ValueError(
                f"TADA stream chunk for {request_id!r} must be "
                f"[frames, {PACKED_FRAME_WIDTH}], got {tuple(codes.shape)}"
            )
        else:
            pass
        return codes.detach().to(dtype=torch.float32)

    def ingest(
        self, request_id: str, state: TadaStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        packed_rows = codes.to(device="cpu", dtype=torch.float32)
        for packed_row in packed_rows.unbind(0):
            frame = packed_row[:LATENT_DIM].contiguous()
            time_before_frames = max(int(packed_row[LATENT_DIM].item()), 1)
            if not state.started:
                state.started = True
                state.expanded_frames.append(frame)
            else:
                silence_frames = time_before_frames - 1
                if silence_frames > 0:
                    state.expanded_frames.extend(
                        [self.silence_frame] * silence_frames
                    )
                else:
                    pass
                state.expanded_frames.append(frame)

    def should_decode(self, state: TadaStreamState, *, is_final: bool) -> bool:
        del is_final
        pending_frames = len(state.expanded_frames) - state.emitted_frames
        return pending_frames >= self.chunk_frames + self.right_context_frames

    def decode_delta(
        self, request_id: str, state: TadaStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        if is_final and not state.trailing_resolved:
            self.resolve_trailing_frames(request_id, state)
        else:
            pass
        plan = build_window_plan(
            total_frames=len(state.expanded_frames),
            emitted_frames=state.emitted_frames,
            chunk_frames=self.chunk_frames,
            left_context_frames=self.left_context_frames,
            right_context_frames=self.right_context_frames,
            is_final=is_final,
        )
        if plan is None:
            return None
        else:
            pass
        return self.decode_window_batch([(request_id, state, plan)])[request_id]

    def select_step_participants(self) -> list[tuple[str, TadaStreamState]]:
        participants: list[tuple[str, TadaStreamState]] = []
        threshold = self.chunk_frames + self.right_context_frames
        for request_id, state in self.stream_state_items():
            if self.is_aborted(request_id):
                continue
            else:
                pass
            if len(state.expanded_frames) - state.emitted_frames >= threshold:
                participants.append((request_id, state))
            else:
                pass
            if len(participants) >= self.max_batch_size:
                break
            else:
                pass
        return participants

    def build_step_plan(
        self, participants: list[tuple[str, TadaStreamState]]
    ) -> list[TadaWindowPlan]:
        plans: list[TadaWindowPlan] = []
        for _, state in participants:
            plan = build_window_plan(
                total_frames=len(state.expanded_frames),
                emitted_frames=state.emitted_frames,
                chunk_frames=self.chunk_frames,
                left_context_frames=self.left_context_frames,
                right_context_frames=self.right_context_frames,
                is_final=False,
            )
            if plan is None:
                raise RuntimeError(
                    "TADA selected a stream without a complete window"
                )
            else:
                pass
            plans.append(plan)
        return plans

    def run_step(
        self,
        participants: list[tuple[str, TadaStreamState]],
        plan: list[TadaWindowPlan],
    ) -> dict[str, torch.Tensor]:
        items = [
            (request_id, state, window_plan)
            for (request_id, state), window_plan in zip(
                participants, plan, strict=True
            )
        ]
        return self.decode_window_batch(items)

    def decode_window_batch(
        self,
        items: list[tuple[str, TadaStreamState, TadaWindowPlan]],
    ) -> dict[str, torch.Tensor]:
        """Decode equal-length windows in one batch and emit each plan's chunk."""
        grouped: dict[int, list[tuple[str, TadaStreamState, TadaWindowPlan]]] = {}
        for item in items:
            plan = item[2]
            window_frames = plan.window_end_frame - plan.window_start_frame
            grouped.setdefault(window_frames, []).append(item)
        audios: dict[str, torch.Tensor] = {}
        for group in grouped.values():
            windows = [
                torch.stack(
                    state.expanded_frames[
                        plan.window_start_frame : plan.window_end_frame
                    ]
                )
                for _, state, plan in group
            ]
            batch = torch.stack(windows).to(device=self.device, dtype=self.dtype)
            token_masks = (batch.abs().sum(dim=-1) != 0).long()
            with self.lock, torch.inference_mode():
                decoded = self.decoder(batch, token_masks=token_masks)
            for row, (request_id, state, plan) in enumerate(group):
                samples = decoded[row, 0].float().cpu()
                start_sample = plan.emit_sample_offset * SAMPLES_PER_FRAME
                audios[request_id] = samples[
                    start_sample : start_sample + plan.emit_frames * SAMPLES_PER_FRAME
                ].contiguous()
                state.emitted_frames += plan.emit_frames
        return audios

    def resolve_trailing_frames(
        self, request_id: str, state: TadaStreamState
    ) -> None:
        state.trailing_resolved = True
        payload = self.stream_payloads.get(request_id)
        if payload is None:
            return
        else:
            pass
        if not state.expanded_frames:
            return
        else:
            pass
        final_state = load_tada_state(payload)
        time_before = final_state.frame_time_before
        if time_before is not None and time_before.numel() > 0:
            state.trailing_frames = max(int(time_before[-1].item()), 0)
        else:
            pass
        if state.trailing_frames > 0:
            state.expanded_frames.extend(
                [self.silence_frame] * state.trailing_frames
            )
        else:
            pass

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: TadaStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        final_state = load_tada_state(payload)
        if final_state.frames is None or final_state.frame_time_before is None:
            return None
        else:
            pass
        return self.decode_frames(final_state.frames, final_state.frame_time_before)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: TadaStreamState
    ) -> Mapping[str, object]:
        del request_id, state
        final_state = load_tada_state(payload)
        final_state.frames = None
        final_state.frame_time_before = None
        final_state.prompt_acoustic = None
        final_state.prompt_acoustic_mask = None
        final_state.prompt_time_before = None
        final_state.prompt_time_after = None
        data: dict[str, object] = dict(final_state.to_dict())
        data["modality"] = "audio"
        data["sample_rate"] = SAMPLE_RATE
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        return data


__all__ = [
    "DECODER_MAX_FRAMES",
    "DEFAULT_CHUNK_FRAMES",
    "DEFAULT_LEFT_CONTEXT_FRAMES",
    "DEFAULT_RIGHT_CONTEXT_FRAMES",
    "LATENT_DIM",
    "PACKED_FRAME_WIDTH",
    "TadaStreamingVocoder",
    "TadaStreamState",
]
