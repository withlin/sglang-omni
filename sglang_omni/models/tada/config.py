# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for HumeAI TADA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.tada"
DEFAULT_LANGUAGE_NAME = "English"
SUPPORTED_LANGUAGES = ("en", "ja", "de", "fr", "es", "ch", "ar", "it", "pl", "pt")
LANGUAGE_TO_CODE = {
    "English": "en",
    "Japanese": "ja",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Chinese": "ch",
    "Arabic": "ar",
    "Italian": "it",
    "Polish": "pl",
    "Portuguese": "pt",
}
ALIGNMENT_CHECK_MODES = ("off", "warn", "reject")
# note: the engine shares one GPU with the reference encoder (~0.6 GB), the
# aligner (~0.9 GB), and the decoder (~0.65 GB); 0.6 fits the 8.2 GB of engine
# weights plus KV on a 16 GB card and leaves room on 24 GB.
TADA_ENGINE_GPU_MEMORY_FRACTION = 0.6


class TadaPreprocessingFactoryArgs(FactoryArgs):
    max_target_tokens: int = Field(default=2048, gt=0)
    num_extra_steps: int = Field(default=0, ge=0)
    max_extra_steps: int = Field(default=512, ge=0)
    default_language: str = DEFAULT_LANGUAGE_NAME
    max_concurrency: int = Field(default=8, ge=1)


class TadaPreprocessingStageConfig(StageConfig):
    factory: TadaPreprocessingFactoryArgs = Field(
        default_factory=TadaPreprocessingFactoryArgs
    )


class TadaReferenceEncodeFactoryArgs(FactoryArgs):
    codec_model_path: str = "HumeAI/tada-codec"
    tokenizer_path: str | None = None
    aligner_cache_size: int = Field(default=2, ge=1)
    encoder_noise_std: float = Field(default=0.5, ge=0)
    max_reference_seconds: float = Field(default=60.0, gt=0)
    use_text_in_prompt: bool = False
    num_transition_steps: int = Field(default=5, ge=0)
    reference_peak_normalize: float = Field(default=0.0, ge=0.0, le=1.0)
    reference_trim_top_db: float | None = None
    alignment_check: str = "warn"
    max_concurrency: int = Field(default=1, ge=1)


class TadaReferenceEncodeStageConfig(StageConfig):
    factory: TadaReferenceEncodeFactoryArgs = Field(
        default_factory=TadaReferenceEncodeFactoryArgs
    )


class TadaEngineFactoryArgs(FactoryArgs):
    max_running_requests: int = Field(default=4, ge=1)
    num_flow_matching_steps: int = Field(default=10, ge=1)
    acoustic_cfg_scale: float = Field(default=1.8, ge=1.0)
    duration_cfg_scale: float = Field(default=1.0, ge=1.0)
    cfg_schedule: str = "cosine"
    time_schedule: str = "logsnr"
    noise_temperature: float = Field(default=0.9, gt=0)
    negative_condition_source: str = "zero"


class TadaEngineStageConfig(EngineStageConfig):
    factory: TadaEngineFactoryArgs = Field(default_factory=TadaEngineFactoryArgs)


class TadaVocoderFactoryArgs(FactoryArgs):
    codec_model_path: str = "HumeAI/tada-codec"
    use_embedded_decoder: bool = False
    stream_chunk_frames: int = Field(default=50, ge=1)
    stream_left_context_frames: int = Field(default=128, ge=0)
    stream_right_context_frames: int = Field(default=32, ge=0)
    stream_max_batch_size: int = Field(default=8, ge=1)


class TadaVocoderStageConfig(StageConfig):
    factory: TadaVocoderFactoryArgs = Field(default_factory=TadaVocoderFactoryArgs)


class TadaPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "TadaForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("tada",)
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_required: ClassVar[bool] = True
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        {"Arabic", "Polish"}
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": TadaPreprocessingStageConfig,
        "reference_encode": TadaReferenceEncodeStageConfig,
        "tts_engine": TadaEngineStageConfig,
        "vocoder": TadaVocoderStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        TadaPreprocessingStageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        TadaReferenceEncodeStageConfig(
            name="reference_encode",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_reference_encode_executor",
            gpu=0,
            next="tts_engine",
        ),
        TadaEngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            gpu=0,
            gpu_memory_fraction=TADA_ENGINE_GPU_MEMORY_FRACTION,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        TadaVocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "preprocessing":
            reference_factory = self.stage_named("reference_encode").factory
            assert isinstance(reference_factory, TadaReferenceEncodeFactoryArgs)
            return {"tokenizer_path": reference_factory.tokenizer_path}
        else:
            pass
        return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = TadaPipelineConfig
