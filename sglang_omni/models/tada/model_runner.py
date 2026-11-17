# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for TADA's per-token acoustic recurrence."""

from __future__ import annotations

import math

import torch
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.tada.components.flow_head import (
    TadaFlowOptions,
    solve_flow_matching,
)
from sglang_omni.models.tada.components.gray_code import decode_gray_bits_to_time
from sglang_omni.models.tada.request_builders import TadaRequestData
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest


def next_token_policy(position: int, schedule_length: int, extra_steps: int) -> str:
    if position + 1 < schedule_length:
        return "schedule"
    elif position < schedule_length - 1 + extra_steps:
        return "sample"
    else:
        pass
    return "finish"


def should_collect_predicted(
    position: int, shift_acoustic: int, prompt_frames: int
) -> bool:
    return position - shift_acoustic >= prompt_frames


class TadaModelRunner(ModelRunner[TadaRequestData]):
    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        flow_options: TadaFlowOptions,
        noise_temperature: float,
        negative_condition_source: str,
        shift_acoustic: int,
        acoustic_dim: int,
        num_time_classes: int,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.flow_options = flow_options
        self.noise_temperature = float(noise_temperature)
        self.negative_condition_source = negative_condition_source
        self.shift_acoustic = int(shift_acoustic)
        self.acoustic_dim = int(acoustic_dim)
        self.num_time_classes = int(num_time_classes)
        self.num_time_bits = math.ceil(math.log2(self.num_time_classes))
        self.time_dim = 2 * self.num_time_bits
        self.latent_dim = self.acoustic_dim + self.time_dim
        self.request_data: dict[str, TadaRequestData] = {}

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        if not requests:
            return
        else:
            pass
        device = forward_batch.input_ids.device
        rows: list[torch.Tensor] = []
        for request in requests:
            data: TadaRequestData = request.data
            if data.current_position > data.prefill_end:
                raise RuntimeError(
                    "TADA v1 does not support KV retraction; raise "
                    "mem_fraction_static or lower max_running_requests"
                )
            else:
                pass
            self.materialize_request(data, device)
            embeddings = self.build_prompt_embeds(data, device)
            rows.append(embeddings[0])
            self.request_data[request.request_id] = data
        forward_batch.input_embeds = torch.cat(rows, dim=0)

    def materialize_request(
        self, data: TadaRequestData, device: torch.device
    ) -> None:
        if data.schedule.device != device:
            data.schedule = data.schedule.to(device)
        else:
            pass
        for field_name in (
            "prompt_acoustic",
            "prompt_acoustic_mask",
            "prompt_time_before",
            "prompt_time_after",
        ):
            tensor = getattr(data, field_name)
            if tensor is not None and tensor.device != device:
                setattr(data, field_name, tensor.to(device))
            else:
                pass

    def build_prompt_embeds(
        self, data: TadaRequestData, device: torch.device
    ) -> torch.Tensor:
        prompt_len = int(data.prefill_end)
        shift = self.shift_acoustic
        token_embeds = self.model.get_input_embeddings()(
            data.schedule[:, :prompt_len]
        )
        dtype = token_embeds.dtype
        acoustic = torch.zeros(
            1,
            prompt_len,
            self.acoustic_dim,
            device=device,
            dtype=dtype,
        )
        acoustic_count = min(prompt_len - shift - 1, data.prompt_acoustic.shape[1])
        if acoustic_count > 0:
            acoustic[:, shift + 1 : shift + 1 + acoustic_count] = (
                data.prompt_acoustic[:, :acoustic_count].to(
                    device=device, dtype=dtype
                )
            )
        else:
            pass
        masks = torch.zeros(1, prompt_len, device=device, dtype=torch.long)
        if acoustic_count > 0:
            masks[:, shift + 1 : shift + 1 + acoustic_count] = (
                data.prompt_acoustic_mask[:, :acoustic_count].to(device=device)
            )
        else:
            pass
        time_before = torch.zeros(1, prompt_len, device=device, dtype=torch.long)
        time_after = torch.zeros(1, prompt_len, device=device, dtype=torch.long)
        time_count = min(
            prompt_len - shift - 1, data.prompt_time_before.shape[1] - 1
        )
        if time_count > 0:
            time_before[:, shift + 1 : shift + 1 + time_count] = (
                data.prompt_time_before[:, 1 : 1 + time_count].to(device=device)
            )
            time_after[:, shift + 1 : shift + 1 + time_count] = (
                data.prompt_time_after[:, 1 : 1 + time_count].to(device=device)
            )
        else:
            pass
        return (
            token_embeds
            + self.model.acoustic_proj(acoustic)
            + self.model.acoustic_mask_emb(masks)
            + self.model.time_start_embed(time_before)
            + self.model.time_end_embed(time_after)
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        if not requests:
            return
        else:
            pass
        needs_sampling = any(
            next_token_policy(
                request.data.current_position,
                request.data.schedule.shape[1],
                request.data.extra_steps,
            )
            == "sample"
            for request in requests
        )
        self.model.compute_text_logits = needs_sampling
        device = forward_batch.input_ids.device
        token_embeds = self.model.get_input_embeddings()(
            forward_batch.input_ids.to(device)
        )
        acoustic = torch.cat(
            [
                request.data.feedback_acoustic.to(
                    device=device, dtype=token_embeds.dtype
                )
                for request in requests
            ],
            dim=0,
        )
        masks = torch.cat(
            [request.data.feedback_mask.to(device=device) for request in requests],
            dim=0,
        )
        time_before = torch.cat(
            [
                request.data.feedback_time_before.to(device=device)
                for request in requests
            ],
            dim=0,
        )
        time_after = torch.cat(
            [
                request.data.feedback_time_after.to(device=device)
                for request in requests
            ],
            dim=0,
        )
        forward_batch.input_embeds = (
            token_embeds
            + self.model.acoustic_proj(acoustic)
            + self.model.acoustic_mask_emb(masks.squeeze(1))
            + self.model.time_start_embed(time_before.squeeze(1))
            + self.model.time_end_embed(time_after.squeeze(1))
        )

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        del schedule_batch, requests
        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        del schedule_batch, requests
        return CaptureHiddenMode.LAST

    def post_prefill(
        self,
        result: object,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        if not requests:
            return
        else:
            pass
        hidden = self.hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        else:
            pass
        offset = 0
        next_token_ids: list[int] = []
        for request in requests:
            data: TadaRequestData = request.data
            length = int(data.req.extend_len)
            request_hidden = hidden[offset : offset + length].unsqueeze(0)
            if request_hidden.shape[1] != length:
                raise RuntimeError("TADA prefill hidden rows are incomplete")
            else:
                pass
            offset += length
            data.prefill_hidden = request_hidden
            self.initialize_recurrence(data)
            schedule_length = int(data.schedule.shape[1])
            if data.prefill_end < schedule_length:
                next_token_ids.append(int(data.schedule[0, data.prefill_end].item()))
            else:
                data.req.finished_reason = FINISH_LENGTH(
                    length=int(data.req.sampling_params.max_new_tokens)
                )
                next_token_ids.append(int(data.schedule[0, -1].item()))
        if offset != hidden.shape[0]:
            raise RuntimeError("TADA prefill hidden rows do not match requests")
        else:
            pass
        result.next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.long, device=hidden.device
        )

    def initialize_recurrence(self, data: TadaRequestData) -> None:
        n_prompt_frames = int(data.prefill_end) - self.shift_acoustic
        if n_prompt_frames > 0:
            data.feedback_acoustic = data.prompt_acoustic[
                :, n_prompt_frames - 1
            ].detach()
            data.feedback_mask = data.prompt_acoustic_mask[
                :, n_prompt_frames - 1 : n_prompt_frames
            ].detach()
            data.feedback_time_before = data.prompt_time_before[
                :, n_prompt_frames : n_prompt_frames + 1
            ].detach()
            data.feedback_time_after = data.prompt_time_after[
                :, n_prompt_frames : n_prompt_frames + 1
            ].detach()
            data.collected_frames = [
                data.prompt_acoustic[0, index].detach()
                for index in range(n_prompt_frames)
            ]
            data.collected_time_before = [
                data.prompt_time_before[0, index + 1].detach().reshape(1)
                for index in range(n_prompt_frames)
            ]
        else:
            data.feedback_acoustic = torch.zeros(
                1, self.acoustic_dim, device=data.schedule.device, dtype=torch.float32
            )
            data.feedback_mask = torch.zeros(
                1, 1, device=data.schedule.device, dtype=torch.long
            )
            data.feedback_time_before = torch.zeros(
                1, 1, device=data.schedule.device, dtype=torch.long
            )
            data.feedback_time_after = data.feedback_time_before.clone()
            data.collected_frames = []
            data.collected_time_before = []
        data.trailing_time_added = False

    def post_decode(
        self,
        result: object,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        if not requests:
            return
        else:
            pass
        hidden = self.hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        else:
            pass
        head_dtype = next(self.model.prediction_head.parameters()).dtype
        hidden = hidden.to(head_dtype)
        conditioning = self.model.bottleneck_proj(hidden)
        if self.negative_condition_source == "prompt":
            negative_rows = []
            for request in requests:
                data: TadaRequestData = request.data
                position = data.current_position % data.prefill_hidden.shape[1]
                negative_rows.append(data.prefill_hidden[:, position])
            negative = self.model.bottleneck_proj(
                torch.stack(negative_rows, dim=0).squeeze(1).to(head_dtype)
            )
        else:
            negative = torch.zeros_like(conditioning)
        noise_rows = []
        for request in requests:
            data: TadaRequestData = request.data
            if data.noise_generator is None:
                generator = torch.Generator(device=hidden.device)
                if data.state.seed is not None:
                    generator.manual_seed(int(data.state.seed))
                else:
                    generator.seed()
                data.noise_generator = generator
            else:
                pass
            noise_rows.append(
                torch.randn(
                    1,
                    self.latent_dim,
                    generator=data.noise_generator,
                    device=hidden.device,
                    dtype=head_dtype,
                )
            )
        noise = torch.cat(noise_rows, dim=0) * self.noise_temperature
        speech = solve_flow_matching(
            prediction_head=self.model.prediction_head,
            speech=noise,
            conditioning=conditioning,
            negative_condition=negative,
            acoustic_dim=self.acoustic_dim,
            options=self.flow_options,
        )
        next_token_ids: list[int | None] = []
        sampled_rows: list[int] = []
        for row, request in enumerate(requests):
            data: TadaRequestData = request.data
            speech_row = speech[row]
            predicted_acoustic = speech_row[: self.acoustic_dim]
            gray_bits = speech_row[self.acoustic_dim :]
            predicted_time_before = decode_gray_bits_to_time(
                gray_bits[: self.num_time_bits], self.num_time_bits
            )
            predicted_time_after = decode_gray_bits_to_time(
                gray_bits[self.num_time_bits :], self.num_time_bits
            )
            position = data.current_position
            if position >= self.shift_acoustic:
                prompt_frame_index = position - self.shift_acoustic
                if prompt_frame_index < data.prompt_acoustic.shape[1]:
                    frame = data.prompt_acoustic[0, prompt_frame_index]
                    mask = data.prompt_acoustic_mask[0, prompt_frame_index]
                else:
                    frame = predicted_acoustic
                    mask = torch.ones(1, device=hidden.device, dtype=torch.long)
                if prompt_frame_index < data.prompt_time_before.shape[1] - 1:
                    time_before = data.prompt_time_before[
                        0, prompt_frame_index + 1
                    ]
                    time_after = data.prompt_time_after[0, prompt_frame_index + 1]
                else:
                    time_before = predicted_time_before
                    time_after = predicted_time_after
                data.collected_frames.append(frame.detach().reshape(-1))
                data.collected_time_before.append(
                    time_before.detach().reshape(1)
                )
                data.feedback_acoustic = frame.detach().reshape(1, -1)
                data.feedback_mask = mask.detach().reshape(1, 1).long()
                data.feedback_time_before = time_before.detach().reshape(1, 1)
                data.feedback_time_after = time_after.detach().reshape(1, 1)
                frame_index = len(data.collected_frames) - 1
                if data.stream and frame_index >= int(data.state.output_start):
                    denormalized = (
                        frame.detach().reshape(-1).float()
                        * float(data.state.acoustic_std)
                        + float(data.state.acoustic_mean)
                    )
                    data.latest_stream_frame = torch.cat(
                        [denormalized, time_before.detach().reshape(1).float()]
                    ).reshape(1, -1)
                else:
                    pass
            else:
                pass
            data.current_position = position + 1
            policy = next_token_policy(
                position, int(data.schedule.shape[1]), int(data.extra_steps)
            )
            if policy == "schedule":
                next_token_ids.append(int(data.schedule[0, position + 1].item()))
            elif policy == "sample":
                next_token_ids.append(None)
                sampled_rows.append(row)
            else:
                next_token_ids.append(int(data.schedule[0, -1].item()))
                data.req.finished_reason = FINISH_LENGTH(
                    length=int(data.req.sampling_params.max_new_tokens)
                )
                if (
                    not data.trailing_time_added
                    and data.feedback_time_before is not None
                ):
                    data.collected_time_before.append(
                        data.feedback_time_before.detach().reshape(1)
                    )
                    data.trailing_time_added = True
                else:
                    pass
        if sampled_rows:
            sampled = self.sample_next_token_ids(
                result.logits_output, forward_batch, schedule_batch, requests
            )
            for row in sampled_rows:
                next_token_ids[row] = int(sampled[row].item())
        else:
            pass
        result.next_token_ids = torch.tensor(
            [int(token) for token in next_token_ids],
            dtype=torch.long,
            device=hidden.device,
        )

    @staticmethod
    def hidden_states(result: object) -> torch.Tensor:
        logits_output = getattr(result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            hidden = getattr(result, "hidden_states", None)
        else:
            pass
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("TADA SGLang forward did not return hidden states")
        else:
            pass
        return hidden

    def on_request_finished(
        self, request_id: str, req_data: TadaRequestData
    ) -> None:
        self.request_data.pop(request_id, None)

    def reset_request(self, request_id: str) -> None:
        self.request_data.pop(request_id, None)


__all__ = [
    "TadaModelRunner",
    "next_token_policy",
    "should_collect_predicted",
]
