# SPDX-License-Identifier: Apache-2.0
"""Pure prompt and schedule construction for TADA."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 50

_PUNCTUATION = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "–": "-",
    "—": "-",
    "―": "-",
    "‐": "-",
    "‑": "-",
    "…": "...",
    "‹": "<",
    "›": ">",
    "«": "<<",
    "»": ">>",
}
_PUNCTUATION_PATTERN = re.compile(
    "|".join(re.escape(character) for character in _PUNCTUATION)
)


@dataclass(frozen=True)
class TadaSpecialTokens:
    bos_id: int
    eos_id: int
    eot_id: int
    pad_id: int
    start_header_id: int
    end_header_id: int


@dataclass(frozen=True)
class TadaTextSchedule:
    input_ids: torch.Tensor
    reference_token_count: int
    target_token_count: int
    header_length: int


@dataclass(frozen=True)
class TadaPromptLayout:
    input_ids: torch.Tensor
    acoustic: torch.Tensor
    acoustic_mask: torch.Tensor
    time_before: torch.Tensor
    time_after: torch.Tensor
    prefill_len: int
    output_start: int


def normalize_text(text: str) -> str:
    """Replace common Unicode punctuation and normalize casing for TADA."""
    text = _PUNCTUATION_PATTERN.sub(
        lambda match: _PUNCTUATION[match.group(0)], text
    )
    text = (
        text.replace("; ", ". ")
        .replace('"', "")
        .replace(":", ",")
        .replace("(", "")
        .replace(")", "")
        .replace("--", "-")
        .replace("-", ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ")
    )
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = re.sub(
        r"([.!?]\s*)(\w)",
        lambda match: match.group(1) + match.group(2).upper(),
        text.lower(),
    )
    if text:
        text = text[0].upper() + text[1:]
    else:
        pass
    return text


def tada_special_tokens_from_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> TadaSpecialTokens:
    return TadaSpecialTokens(
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        eot_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        pad_id=tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
        start_header_id=tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
        end_header_id=tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
    )


def tada_header_text(
    system_prompt: str | None, user_turn_prompt: str | None
) -> str:
    header = (
        "<|start_header_id|>system<|end_header_id|>"
        + (system_prompt or "")
        + "<|eot_id|>"
    )
    if user_turn_prompt:
        header += (
            "<|start_header_id|>user<|end_header_id|>"
            + user_turn_prompt
            + "<|eot_id|>"
        )
    else:
        pass
    header += "<|start_header_id|>assistant<|end_header_id|>"
    return header


def tada_header_length(tokenizer: PreTrainedTokenizerBase) -> int:
    return len(
        tokenizer.encode(
            tada_header_text(None, None), add_special_tokens=False
        )
    )


def build_text_schedule(
    *,
    tokenizer: PreTrainedTokenizerBase,
    reference_text: str,
    target_text: str,
    shift_acoustic: int,
    system_prompt: str | None,
    user_turn_prompt: str | None,
) -> TadaTextSchedule:
    specials = tada_special_tokens_from_tokenizer(tokenizer)
    header_ids = tokenizer.encode(
        tada_header_text(system_prompt, user_turn_prompt),
        add_special_tokens=False,
    )
    reference_ids = tokenizer.encode(
        normalize_text(reference_text), add_special_tokens=False
    )
    target_ids = tokenizer.encode(
        normalize_text(target_text), add_special_tokens=False
    )
    input_ids = (
        [specials.bos_id]
        + header_ids
        + reference_ids
        + target_ids
        + [specials.eot_id] * shift_acoustic
    )
    return TadaTextSchedule(
        input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        reference_token_count=len(reference_ids),
        target_token_count=len(target_ids),
        header_length=len(header_ids),
    )


def build_prompt_layout(
    *,
    specials: TadaSpecialTokens,
    shift_acoustic: int,
    num_transition_steps: int,
    num_time_classes: int,
    acoustic_dim: int,
    input_ids: torch.Tensor,
    reference_token_positions: torch.Tensor,
    reference_token_values: torch.Tensor,
    reference_audio_length: torch.Tensor,
    header_length: int,
    use_text_in_prompt: bool,
) -> TadaPromptLayout:
    del acoustic_dim
    token_count = int(input_ids.shape[-1])
    audio_feature_count = (
        reference_audio_length.to(torch.float32) / SAMPLE_RATE * FRAMES_PER_SECOND
    ).ceil().long()
    selected = reference_token_positions.to(torch.long).clone()
    ending_index = token_count - shift_acoustic - 1 - header_length
    if 0 <= ending_index < selected.shape[-1]:
        selected[..., ending_index] = audio_feature_count.to(selected.device)
    else:
        pass
    previous = F.pad(selected, (1, 0), value=1)
    gaps = (selected - previous[..., :-1]).clamp(
        min=0, max=num_time_classes - 1
    )
    gaps = F.pad(gaps, (1, 0), value=0)
    time_before = gaps[..., :-1]
    time_after = gaps[..., 1:]
    values = reference_token_values.unsqueeze(0)
    acoustic_mask = torch.ones(values.shape[:2], dtype=torch.long)
    values = F.pad(values, (0, 0, header_length, 0))
    acoustic_mask = F.pad(acoustic_mask, (header_length, 0))
    time_before = F.pad(time_before, (header_length, 0))
    time_after = F.pad(time_after, (header_length, 0))
    if num_transition_steps > 0:
        values = values[:, :-num_transition_steps]
        acoustic_mask = acoustic_mask[:, :-num_transition_steps]
        time_before = time_before[:, :-num_transition_steps]
        time_after = time_after[:, :-num_transition_steps]
    else:
        pass
    prompt_len = int(values.shape[1])
    if prompt_len <= 0:
        raise ValueError(
            "TADA reference prompt is shorter than num_transition_steps; "
            "use a longer reference or a smaller transition"
        )
    else:
        pass
    acoustic_mask = torch.cat(
        [acoustic_mask[:, 1:], torch.ones_like(acoustic_mask[:, :1])], dim=-1
    )
    masked_input_ids = input_ids.clone()
    if use_text_in_prompt:
        pass
    else:
        in_header = False
        for position in range(prompt_len):
            token = int(masked_input_ids[0, position].item())
            if token == specials.start_header_id:
                in_header = True
                keep = True
            elif token == specials.end_header_id:
                in_header = False
                keep = True
            elif in_header:
                keep = True
            elif token in (
                specials.eot_id,
                specials.bos_id,
                specials.eos_id,
            ):
                keep = True
            else:
                keep = False
            if not keep:
                masked_input_ids[0, position] = specials.pad_id
            else:
                pass
    n_acoustic = min(token_count - shift_acoustic - 1, prompt_len - 1)
    n_time = min(token_count - shift_acoustic - 1, time_before.shape[1] - 1)
    frame_cap = max(0, time_before.shape[1] - 2)
    if n_acoustic > 0 and n_time > 0:
        n_frames = min(n_acoustic, n_time, frame_cap)
    else:
        n_frames = 0
    if n_frames > 0:
        prefill_len = min(token_count, shift_acoustic + n_frames + 1)
    else:
        prefill_len = 0
    output_start = prompt_len + num_transition_steps - 1
    return TadaPromptLayout(
        input_ids=masked_input_ids,
        acoustic=values,
        acoustic_mask=acoustic_mask,
        time_before=time_before,
        time_after=time_after,
        prefill_len=prefill_len,
        output_start=output_start,
    )


__all__ = [
    "SAMPLE_RATE",
    "FRAMES_PER_SECOND",
    "TadaPromptLayout",
    "TadaSpecialTokens",
    "TadaTextSchedule",
    "build_prompt_layout",
    "build_text_schedule",
    "normalize_text",
    "tada_header_length",
    "tada_header_text",
    "tada_special_tokens_from_tokenizer",
]
