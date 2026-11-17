# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the TADA pipeline."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig

from sglang_omni.models.tada.components.codec_aligner import (
    TadaAligner,
    aligner_subfolder,
)
from sglang_omni.models.tada.components.codec_decoder import TadaDecoder
from sglang_omni.models.tada.components.codec_encoder import TadaEncoder
from sglang_omni.models.tada.config import (
    ALIGNMENT_CHECK_MODES,
    LANGUAGE_TO_CODE,
)
from sglang_omni.models.tada.engine_builder import TadaEngineBuilder
from sglang_omni.models.tada.payload_types import (
    TadaState,
    load_tada_state,
    store_tada_state,
)
from sglang_omni.models.tada.prompt import (
    FRAMES_PER_SECOND,
    SAMPLE_RATE,
    build_prompt_layout,
    build_text_schedule,
    normalize_text,
    tada_header_length,
    tada_special_tokens_from_tokenizer,
)
from sglang_omni.models.tada.streaming_vocoder import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_LEFT_CONTEXT_FRAMES,
    DEFAULT_RIGHT_CONTEXT_FRAMES,
    DEFAULT_STREAM_MAX_BATCH_SIZE,
    TadaStreamingVocoder,
)
from sglang_omni.models.weight_loader import (
    load_weights_by_prefix,
    resolve_model_path,
)
from sglang_omni.preprocessing.cache_key import (
    hash_bytes,
    reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


def first_not_none(*values: object, default: object = None) -> object:
    return next((value for value in values if value is not None), default)


def as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def reference_path(value: object) -> str | None:
    if value is None:
        return None
    elif isinstance(value, str):
        return value
    elif isinstance(value, dict):
        path = (
            value.get("audio_path")
            or value.get("path")
            or audio_data_uri_from_reference(value)
        )
        return str(path) if path else None
    else:
        pass
    raise TypeError("TADA reference audio must be a path or data URI")


def request_inputs(payload: StagePayload) -> dict[str, object]:
    value = payload.request.inputs
    if isinstance(value, str):
        return {"text": value}
    else:
        pass
    return as_dict(value)


def request_params(payload: StagePayload) -> dict[str, object]:
    return as_dict(payload.request.params)


def request_tts_params(payload: StagePayload) -> dict[str, object]:
    return as_dict(as_dict(payload.request.metadata).get("tts_params"))


def resolve_tokenizer_path(
    tokenizer_path: str | None, checkpoint_dir: Path
) -> str:
    if tokenizer_path:
        return tokenizer_path
    else:
        pass
    if checkpoint_dir.exists():
        for filename in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
            if (checkpoint_dir / filename).exists():
                return str(checkpoint_dir)
            else:
                pass
    else:
        pass
    raise ValueError(
        "TADA requires tokenizer_path; the Llama 3.2 tokenizer is not shipped "
        "with the weights"
    )


def model_config_from_path(model_path: str) -> LlamaConfig:
    return LlamaConfig.from_pretrained(model_path)


def load_codec_module(
    module: torch.nn.Module, directory: Path, device: torch.device
) -> torch.nn.Module:
    """Copy-load codec weights; assign=True is unsafe with weight-norm parametrizations."""
    state_dict = load_weights_by_prefix(str(directory), prefix="")
    module.load_state_dict(state_dict, strict=True)
    return module.to(device=device, dtype=torch.bfloat16).eval()


def create_preprocessing_executor(
    model_path: str,
    *,
    tokenizer_path: str | None = None,
    num_extra_steps: int = 0,
    max_extra_steps: int = 512,
    max_target_tokens: int = 2048,
    default_language: str = "English",
    max_concurrency: int = 8,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if default_language not in LANGUAGE_TO_CODE:
        raise ValueError(
            f"TADA default_language must be one of {sorted(LANGUAGE_TO_CODE)}; "
            f"got {default_language!r}"
        )
    else:
        pass
    config = model_config_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_path(tokenizer_path, Path(model_path))
    )

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = request_inputs(payload)
        params = request_params(payload)
        tts_params = request_tts_params(payload)
        references = inputs.get("references")
        if isinstance(references, list) and len(references) > 1:
            raise ValueError("TADA accepts exactly one reference audio")
        else:
            pass
        reference = (
            references[0]
            if isinstance(references, list)
            and references
            and isinstance(references[0], dict)
            else {}
        )
        text = str(
            first_not_none(
                inputs.get("input"), inputs.get("text"), default=""
            )
        )
        if not text.strip():
            raise ValueError("TADA requires non-empty input text")
        else:
            pass
        reference_audio = first_not_none(
            reference_path(inputs.get("ref_audio")),
            reference_path(tts_params.get("ref_audio")),
            reference_path(reference),
        )
        reference_text_value = first_not_none(
            inputs.get("ref_text"),
            tts_params.get("ref_text"),
            reference.get("text"),
        )
        if not reference_audio or not reference_text_value:
            raise ValueError("TADA requires a reference audio and its transcript")
        else:
            pass
        language_name = str(
            first_not_none(
                inputs.get("language"),
                params.get("language"),
                tts_params.get("language"),
                default=default_language,
            )
        )
        language = LANGUAGE_TO_CODE.get(language_name)
        if language is None:
            raise ValueError(
                f"TADA supports {sorted(LANGUAGE_TO_CODE)}; got {language_name!r}"
            )
        else:
            pass
        schedule = build_text_schedule(
            tokenizer=tokenizer,
            reference_text=str(reference_text_value),
            target_text=text,
            shift_acoustic=int(config.shift_acoustic),
            system_prompt=None,
            user_turn_prompt=None,
        )
        if schedule.target_token_count > max_target_tokens:
            raise ValueError(
                f"TADA target text has {schedule.target_token_count} tokens, "
                f"over the {max_target_tokens} limit"
            )
        else:
            pass
        extra_steps = int(
            first_not_none(
                inputs.get("num_extra_steps"),
                tts_params.get("num_extra_steps"),
                default=num_extra_steps,
            )
        )
        if extra_steps < 0 or extra_steps > max_extra_steps:
            raise ValueError(
                f"TADA num_extra_steps must be in [0, {max_extra_steps}]"
            )
        else:
            pass
        seed_value = first_not_none(
            inputs.get("seed"), params.get("seed"), tts_params.get("seed")
        )
        specials = tada_special_tokens_from_tokenizer(tokenizer)
        state = TadaState(
            input_text=text,
            reference_text=str(reference_text_value),
            reference_audio=str(reference_audio),
            language=language,
            input_ids=schedule.input_ids[0].tolist(),
            num_extra_steps=extra_steps,
            seed=None if seed_value is None else int(seed_value),
            acoustic_mean=float(config.acoustic_mean),
            acoustic_std=float(config.acoustic_std),
            vocab_size=int(config.vocab_size),
            pad_token_id=int(specials.pad_id),
            sample_rate=SAMPLE_RATE,
        )
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


@dataclass(frozen=True)
class TadaReferenceItem:
    audio: str
    transcript: str
    language: str


@dataclass(frozen=True)
class TadaReferenceArtifact:
    token_values: torch.Tensor
    token_positions: torch.Tensor
    token_masks: torch.Tensor
    audio_length: torch.Tensor
    header_length: int


class TadaAlignerCache:
    def __init__(
        self,
        codec_dir: Path,
        tokenizer: AutoTokenizer,
        device: torch.device,
        capacity: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("TADA aligner cache capacity must be positive")
        else:
            pass
        self.codec_dir = codec_dir
        self.tokenizer = tokenizer
        self.device = device
        self.capacity = int(capacity)
        self.aligners: dict[str, TadaAligner] = {}
        self.lock = threading.Lock()

    def get(self, language: str) -> TadaAligner:
        with self.lock:
            if language in self.aligners:
                aligner = self.aligners.pop(language)
                self.aligners[language] = aligner
                return aligner
            else:
                pass
            while len(self.aligners) >= self.capacity:
                oldest_language, _ = next(iter(self.aligners.items()))
                self.aligners.pop(oldest_language)
            aligner = TadaAligner(tokenizer=self.tokenizer)
            aligner = load_codec_module(
                aligner,
                self.codec_dir / aligner_subfolder(language),
                self.device,
            )
            self.aligners[language] = aligner
            return aligner

    def close(self) -> None:
        with self.lock:
            self.aligners.clear()


class TadaReferenceEncodeHook(
    KeyedReferenceEncodeHook[
        TadaReferenceItem, TadaReferenceArtifact, TadaReferenceArtifact
    ]
):
    def __init__(
        self,
        *,
        checkpoint_id: str,
        encoder: TadaEncoder,
        aligners: TadaAlignerCache,
        noise_std: float,
        max_reference_seconds: float,
        peak_normalize: float,
        trim_top_db: float | None,
        alignment_check: str,
    ) -> None:
        if alignment_check not in ALIGNMENT_CHECK_MODES:
            raise ValueError(
                f"TADA alignment_check must be one of {ALIGNMENT_CHECK_MODES}"
            )
        else:
            pass
        self.model_id = checkpoint_id
        self.model_revision = ""
        self.encoder_id = "tada-encoder"
        self.encoder_config_hash = (
            f"{noise_std}:{FRAMES_PER_SECOND}:{peak_normalize}:{trim_top_db}"
        )
        self.artifact_kind = "tada-prompt"
        self.encoder = encoder
        self.aligners = aligners
        self.noise_std = float(noise_std)
        self.max_reference_seconds = float(max_reference_seconds)
        self.peak_normalize = float(peak_normalize)
        self.trim_top_db = trim_top_db
        self.alignment_check = alignment_check
        self.header_length = tada_header_length(aligners.tokenizer)
        self.device = next(encoder.parameters()).device

    def input_key(self, item: TadaReferenceItem) -> str | None:
        if item.audio.startswith(("data:", "http://", "https://", "file://")):
            return hash_bytes(item.audio.encode("utf-8"))
        else:
            pass
        path = Path(item.audio)
        if path.exists():
            return reference_path_cache_key(str(path))
        else:
            pass
        return hash_bytes(item.audio.encode("utf-8"))

    def options_key(self, item: TadaReferenceItem) -> str:
        transcript_hash = hashlib.sha256(
            item.transcript.encode("utf-8")
        ).hexdigest()
        return (
            f"{item.language}:{transcript_hash}:{self.noise_std}:"
            f"{self.peak_normalize}:{self.trim_top_db}"
        )

    def encode_one(self, item: TadaReferenceItem) -> TadaReferenceArtifact:
        waveform = load_audio(
            item.audio,
            source_name="TADA reference",
            target_sample_rate=SAMPLE_RATE,
            mono=True,
            trim_top_db=self.trim_top_db,
            resample_kwargs={
                "lowpass_filter_width": 64,
                "rolloff": 0.95,
                "resampling_method": "sinc_interp_kaiser",
            },
        )
        if self.peak_normalize > 0:
            peak = float(abs(waveform).max())
            if peak > 0:
                waveform = waveform * (self.peak_normalize / peak)
            else:
                pass
        else:
            pass
        audio = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        duration = audio.shape[-1] / SAMPLE_RATE
        if duration <= 0 or duration > self.max_reference_seconds:
            raise ValueError(
                f"TADA reference audio must be 0-{self.max_reference_seconds} "
                f"seconds; got {duration:.2f}"
            )
        else:
            pass
        audio = audio.to(self.device)
        aligner = self.aligners.get(item.language)
        text_tokens = aligner.tokenizer.encode(
            normalize_text(item.transcript),
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        audio_length = torch.tensor(
            [audio.shape[-1]], device=self.device, dtype=torch.long
        )
        alignment = aligner(
            audio,
            text_tokens=text_tokens,
            audio_length=audio_length,
            sample_rate=SAMPLE_RATE,
        )
        if self.alignment_check == "off":
            pass
        elif self.alignment_check == "warn":
            for message in alignment.report.messages:
                logger.warning(
                    f"TADA alignment warning for language {item.language}: {message}"
                )
        else:
            if alignment.report.has_problems:
                raise ValueError(
                    f"TADA reference alignment failed for language "
                    f"{item.language}: {'; '.join(alignment.report.messages)}"
                )
            else:
                pass
        encoder_output = self.encoder(
            audio,
            text_token_positions=alignment.token_positions.to(self.device),
            token_masks=alignment.token_masks.to(self.device),
            sample=self.noise_std > 0,
            noise_std=self.noise_std,
        )
        return TadaReferenceArtifact(
            token_values=encoder_output.token_values[0].detach(),
            token_positions=alignment.token_positions.detach(),
            token_masks=alignment.token_masks.detach(),
            audio_length=audio_length.detach(),
            header_length=self.header_length,
        )

    def store_artifact(self, artifact: TadaReferenceArtifact) -> TadaReferenceArtifact:
        return TadaReferenceArtifact(
            token_values=artifact.token_values.detach().cpu(),
            token_positions=artifact.token_positions.detach().cpu(),
            token_masks=artifact.token_masks.detach().cpu(),
            audio_length=artifact.audio_length.detach().cpu(),
            header_length=artifact.header_length,
        )

    def load_artifact(self, stored: TadaReferenceArtifact) -> TadaReferenceArtifact:
        return TadaReferenceArtifact(
            token_values=stored.token_values.to(self.device),
            token_positions=stored.token_positions.to(self.device),
            token_masks=stored.token_masks.to(self.device),
            audio_length=stored.audio_length.to(self.device),
            header_length=stored.header_length,
        )


def create_reference_encode_executor(
    model_path: str,
    *,
    codec_model_path: str = "HumeAI/tada-codec",
    tokenizer_path: str | None = None,
    aligner_cache_size: int = 2,
    encoder_noise_std: float = 0.5,
    max_reference_seconds: float = 60.0,
    use_text_in_prompt: bool = False,
    num_transition_steps: int = 5,
    alignment_check: str = "warn",
    reference_peak_normalize: float = 0.0,
    reference_trim_top_db: float | None = None,
    max_concurrency: int = 1,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    concrete_device = resolve_concrete_device(device, gpu_id)
    checkpoint_dir = resolve_model_path(model_path)
    config = LlamaConfig.from_pretrained(checkpoint_dir)
    codec_dir = resolve_model_path(codec_model_path)
    tokenizer_root = resolve_tokenizer_path(tokenizer_path, checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root)
    encoder = load_codec_module(
        TadaEncoder(
            acoustic_mean=float(config.acoustic_mean),
            acoustic_std=float(config.acoustic_std),
        ),
        Path(codec_dir) / "encoder",
        concrete_device,
    )
    aligners = TadaAlignerCache(
        Path(codec_dir), tokenizer, concrete_device, aligner_cache_size
    )
    hook = TadaReferenceEncodeHook(
        checkpoint_id=str(model_path),
        encoder=encoder,
        aligners=aligners,
        noise_std=encoder_noise_std,
        max_reference_seconds=max_reference_seconds,
        peak_normalize=reference_peak_normalize,
        trim_top_db=reference_trim_top_db,
        alignment_check=alignment_check,
    )
    service = ReferenceEncodeService(
        hook,
        max_items=16,
        max_bytes=256 * 1024 * 1024,
        log_prefix="tada",
    )
    specials = tada_special_tokens_from_tokenizer(tokenizer)

    def _encode(payload: StagePayload) -> StagePayload:
        state = load_tada_state(payload)
        if state.reference_audio is None or not state.reference_text:
            raise ValueError("TADA reference encoding requires audio and transcript")
        else:
            pass
        item = TadaReferenceItem(
            audio=state.reference_audio,
            transcript=state.reference_text,
            language=state.language,
        )
        artifact = service.get_or_encode(
            item, desc=f"tada reference {payload.request_id}"
        )
        layout = build_prompt_layout(
            specials=specials,
            shift_acoustic=int(config.shift_acoustic),
            num_transition_steps=int(num_transition_steps),
            num_time_classes=int(config.num_time_classes),
            acoustic_dim=int(config.acoustic_dim),
            input_ids=torch.tensor(state.input_ids, dtype=torch.long).unsqueeze(0),
            reference_token_positions=artifact.token_positions,
            reference_token_values=artifact.token_values,
            reference_audio_length=artifact.audio_length,
            header_length=artifact.header_length,
            use_text_in_prompt=use_text_in_prompt,
        )
        state.input_ids = layout.input_ids[0].tolist()
        state.prompt_acoustic = layout.acoustic[0].cpu()
        state.prompt_acoustic_mask = layout.acoustic_mask[0].cpu()
        state.prompt_time_before = layout.time_before[0].cpu()
        state.prompt_time_after = layout.time_after[0].cpu()
        state.prefill_len = layout.prefill_len
        state.output_start = layout.output_start
        store_tada_state(payload, state)
        return payload

    return SimpleScheduler(
        _encode,
        max_concurrency=max_concurrency,
        shutdown_callback=aligners.close,
    )


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    precision: str = "bfloat16",
    server_args_overrides: dict[str, object] | None = None,
    num_flow_matching_steps: int = 10,
    acoustic_cfg_scale: float = 1.8,
    duration_cfg_scale: float = 1.0,
    cfg_schedule: str = "cosine",
    time_schedule: str = "logsnr",
    noise_temperature: float = 0.9,
    negative_condition_source: str = "zero",
    max_running_requests: int = 4,
    total_gpu_memory_fraction: float | None = None,
):
    builder = TadaEngineBuilder(
        flow_steps=num_flow_matching_steps,
        acoustic_cfg_scale=acoustic_cfg_scale,
        duration_cfg_scale=duration_cfg_scale,
        cfg_schedule=cfg_schedule,
        time_schedule=time_schedule,
        noise_temperature=noise_temperature,
        negative_condition_source=negative_condition_source,
        max_running_requests=max_running_requests,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=precision,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    codec_model_path: str = "HumeAI/tada-codec",
    use_embedded_decoder: bool = False,
    stream_chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    stream_left_context_frames: int = DEFAULT_LEFT_CONTEXT_FRAMES,
    stream_right_context_frames: int = DEFAULT_RIGHT_CONTEXT_FRAMES,
    stream_max_batch_size: int = DEFAULT_STREAM_MAX_BATCH_SIZE,
    device: str | None = None,
    gpu_id: int | None = None,
) -> TadaStreamingVocoder:
    concrete_device = resolve_concrete_device(device, gpu_id)
    decoder = TadaDecoder()
    if use_embedded_decoder:
        state_dict = load_weights_by_prefix(model_path, prefix="_decoder.")
        decoder.load_state_dict(state_dict, strict=True)
        decoder = decoder.to(device=concrete_device, dtype=torch.bfloat16).eval()
    else:
        codec_dir = resolve_model_path(codec_model_path)
        decoder = load_codec_module(
            decoder, Path(codec_dir) / "decoder", concrete_device
        )
    return TadaStreamingVocoder(
        decoder,
        chunk_frames=stream_chunk_frames,
        left_context_frames=stream_left_context_frames,
        right_context_frames=stream_right_context_frames,
        max_batch_size=stream_max_batch_size,
    )


__all__ = [
    "TadaAlignerCache",
    "TadaReferenceArtifact",
    "TadaReferenceEncodeHook",
    "TadaReferenceItem",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_tts_engine_executor",
    "create_vocoder_executor",
    "load_codec_module",
]
