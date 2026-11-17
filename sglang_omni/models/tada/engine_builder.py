# SPDX-License-Identifier: Apache-2.0
"""TADA SGLang engine builder."""

from __future__ import annotations

from collections.abc import Callable

from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.tada.components.flow_head import TadaFlowOptions
from sglang_omni.models.tada.model_runner import TadaModelRunner
from sglang_omni.models.tada.request_builders import (
    TadaRequestData,
    apply_tada_result,
    build_stream_output,
    build_tada_request,
)
from sglang_omni.models.tada.sglang_model import TadaSGLangModel
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

DEFAULT_MEM_FRACTION_STATIC = 0.6


class TadaEngineBuilder(TtsEngineBuilder[TadaRequestData]):
    model_name = "TADA"
    context_length = 8192

    def __init__(
        self,
        *,
        flow_steps: int = 10,
        acoustic_cfg_scale: float = 1.8,
        duration_cfg_scale: float = 1.0,
        cfg_schedule: str = "cosine",
        time_schedule: str = "logsnr",
        noise_temperature: float = 0.9,
        negative_condition_source: str = "zero",
        max_running_requests: int = 4,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        if total_gpu_memory_fraction is not None and not (
            0.0 < total_gpu_memory_fraction < 1.0
        ):
            raise ValueError("TADA total_gpu_memory_fraction must be in (0, 1)")
        else:
            pass
        if flow_steps < 1:
            raise ValueError("TADA num_flow_matching_steps must be positive")
        else:
            pass
        if negative_condition_source not in {"zero", "prompt"}:
            raise ValueError(
                "TADA negative_condition_source must be 'zero' or 'prompt'"
            )
        else:
            pass
        if cfg_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(f"Unsupported TADA cfg_schedule {cfg_schedule!r}")
        else:
            pass
        if time_schedule not in {"uniform", "cosine", "logsnr"}:
            raise ValueError(f"Unsupported TADA time_schedule {time_schedule!r}")
        else:
            pass
        self.model_arch_override = "TadaForCausalLM"
        self.flow_options = TadaFlowOptions(
            num_flow_matching_steps=int(flow_steps),
            acoustic_cfg_scale=float(acoustic_cfg_scale),
            duration_cfg_scale=float(duration_cfg_scale),
            cfg_schedule=cfg_schedule,
            time_schedule=time_schedule,
        )
        self.noise_temperature = float(noise_temperature)
        self.negative_condition_source = negative_condition_source
        self.max_running_requests = int(max_running_requests)
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.model_runner: TadaModelRunner | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": (
                self.total_gpu_memory_fraction
                if self.total_gpu_memory_fraction is not None
                else DEFAULT_MEM_FRACTION_STATIC
            ),
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("TADA does not implement tensor parallelism")
        else:
            pass
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError("TADA disables SGLang torch compile")
        else:
            pass
        if not bool(overrides.get("disable_cuda_graph", True)):
            raise ValueError("TADA requires eager decode in v1")
        else:
            pass
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        model = model_worker.model_runner.model
        model.eval()
        model.compute_text_logits = False

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> TadaModelRunner:
        model = model_worker.model_runner.model
        config = model.config
        self.model_runner = TadaModelRunner(
            model_worker,
            output_proc,
            flow_options=self.flow_options,
            noise_temperature=self.noise_temperature,
            negative_condition_source=self.negative_condition_source,
            shift_acoustic=int(config.shift_acoustic),
            acoustic_dim=int(config.acoustic_dim),
            num_time_classes=int(config.num_time_classes),
        )
        return self.model_runner

    def make_adapters(
        self, model: TadaSGLangModel | None
    ) -> tuple[
        Callable[[StagePayload], TadaRequestData],
        Callable[[TadaRequestData], StagePayload],
    ]:
        del model
        return build_tada_request, apply_tada_result

    def make_abort_callback(self) -> Callable[[str], None]:
        assert self.model_runner is not None
        return self.model_runner.reset_request

    def extra_scheduler_kwargs(self) -> SchedulerExtras[TadaRequestData]:
        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": False,
        }


__all__ = ["TadaEngineBuilder"]
