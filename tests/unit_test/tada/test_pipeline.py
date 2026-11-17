# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.models.tada import CAPABILITIES
from sglang_omni.models.tada.config import LANGUAGE_TO_CODE, TadaPipelineConfig
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator


def make_validator() -> SpeechRequestValidator:
    config = TadaPipelineConfig(model_path="HumeAI/tada-3b-ml")
    return SpeechRequestValidator(
        default_model=config.model_path,
        required_speech_reference_count=config.required_speech_reference_count,
        speech_reference_text_required=config.speech_reference_text_required,
        additional_speech_languages=config.additional_speech_languages,
    )


def test_tada_accepts_arabic_at_the_public_boundary() -> None:
    validator = make_validator()
    request = validator.parse_generation_request(
        {
            "input": "hello",
            "language": "Arabic",
            "ref_audio": "data:audio/wav;base64,UklGRg==",
            "ref_text": "reference",
        }
    )
    assert request.request.language == "Arabic"


def test_language_map_excludes_unsupported_base_languages() -> None:
    assert LANGUAGE_TO_CODE["Chinese"] == "ch"
    assert LANGUAGE_TO_CODE["Polish"] == "pl"
    assert "Korean" not in LANGUAGE_TO_CODE
    assert "Russian" not in LANGUAGE_TO_CODE


def test_reference_transcript_is_required() -> None:
    validator = make_validator()
    with pytest.raises(SpeechAPIError, match="reference transcript is required"):
        validator.parse_generation_request(
            {
                "input": "hello",
                "ref_audio": "data:audio/wav;base64,UklGRg==",
            }
        )


def test_tada_wires_the_native_streaming_vocoder() -> None:
    config = TadaPipelineConfig(model_path="HumeAI/tada-3b-ml")
    engine_stage = config.stage_named("tts_engine")
    vocoder_stage = config.stage_named("vocoder")
    assert engine_stage.stream_to == ["vocoder"]
    assert vocoder_stage.can_accept_stream_before_payload is True
    assert CAPABILITIES.supports_streaming_vocoder is True
