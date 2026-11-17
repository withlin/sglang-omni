# SPDX-License-Identifier: Apache-2.0
"""Forced aligner and alignment health checks for TADA reference prompts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from sglang_omni.models.tada.config import SUPPORTED_LANGUAGES

ALIGNMENT_MAX_GAP_FRAMES = 150
ALIGNMENT_BUNCHING_WINDOW = 4
ALIGNMENT_BUNCHING_SPAN_FRAMES = 3
ALIGNMENT_MIN_FRAMES_PER_TOKEN = 1.0
ALIGNMENT_MAX_FRAMES_PER_TOKEN = 100.0


def wav2vec2_large_config(vocab_size: int) -> Wav2Vec2Config:
    """Reconstruct the facebook/wav2vec2-large config without a network fetch."""
    return Wav2Vec2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        vocab_size=vocab_size,
    )


@dataclass(frozen=True)
class TadaAlignmentReport:
    is_monotonic: bool
    is_out_of_range: bool
    has_bunching: bool
    has_long_gap: bool
    frames_per_token: float
    messages: tuple[str, ...]

    @property
    def has_problems(self) -> bool:
        return (
            not self.is_monotonic
            or self.is_out_of_range
            or self.has_bunching
            or self.has_long_gap
        )


@dataclass
class TadaAlignOutput:
    token_positions: torch.Tensor
    token_masks: torch.Tensor
    report: TadaAlignmentReport
    logits: torch.Tensor | None = None


def check_alignment(
    token_positions: torch.Tensor, audio_frames: int
) -> TadaAlignmentReport:
    """Apply the paper's alignment anomaly rules to one reference."""
    positions = token_positions.reshape(-1).to(dtype=torch.long, device="cpu")
    messages: list[str] = []
    if positions.numel() == 0:
        return TadaAlignmentReport(
            is_monotonic=True,
            is_out_of_range=False,
            has_bunching=False,
            has_long_gap=False,
            frames_per_token=0.0,
            messages=("reference transcript produced no aligned tokens",),
        )
    else:
        pass
    differences = positions[1:] - positions[:-1]
    is_monotonic = bool((differences >= 1).all())
    if not is_monotonic:
        messages.append("token positions are not strictly increasing")
    else:
        pass
    is_out_of_range = bool(
        positions[0].item() < 1 or positions[-1].item() > int(audio_frames)
    )
    if is_out_of_range:
        messages.append(
            f"token positions [{positions[0].item()}, {positions[-1].item()}] "
            f"fall outside the audio frame range [1, {int(audio_frames)}]"
        )
    else:
        pass
    has_bunching = False
    if positions.numel() >= ALIGNMENT_BUNCHING_WINDOW:
        windows = positions
        spans = (
            windows[ALIGNMENT_BUNCHING_WINDOW - 1 :]
            - windows[: -(ALIGNMENT_BUNCHING_WINDOW - 1)]
        )
        has_bunching = bool((spans <= ALIGNMENT_BUNCHING_SPAN_FRAMES).any())
    else:
        pass
    if has_bunching:
        messages.append(
            "more than three consecutive tokens align within three frames"
        )
    else:
        pass
    has_long_gap = (
        bool((differences > ALIGNMENT_MAX_GAP_FRAMES).any())
        if differences.numel()
        else False
    )
    if has_long_gap:
        messages.append(
            f"an aligned token gap exceeds {ALIGNMENT_MAX_GAP_FRAMES} frames (3 s)"
        )
    else:
        pass
    frames_per_token = float(int(audio_frames)) / float(positions.numel())
    if (
        not ALIGNMENT_MIN_FRAMES_PER_TOKEN
        <= frames_per_token
        <= ALIGNMENT_MAX_FRAMES_PER_TOKEN
    ):
        messages.append(
            f"frames per token {frames_per_token:.2f} is outside "
            f"[{ALIGNMENT_MIN_FRAMES_PER_TOKEN}, {ALIGNMENT_MAX_FRAMES_PER_TOKEN}]"
        )
    else:
        pass
    return TadaAlignmentReport(
        is_monotonic=is_monotonic,
        is_out_of_range=is_out_of_range,
        has_bunching=has_bunching,
        has_long_gap=has_long_gap,
        frames_per_token=frames_per_token,
        messages=tuple(messages),
    )


def align_text_tokens(
    probs: torch.Tensor, text_tokens: torch.Tensor
) -> list[int]:
    """Dynamic-programming forced alignment of tokens to probability frames."""
    frame_count, vocab_size = probs.shape
    token_count = len(text_tokens)
    device = probs.device
    scores = torch.full((frame_count, token_count), -float("inf"), device=device)
    backpointer = torch.zeros(frame_count, token_count, dtype=torch.long, device=device)
    token_probs = probs[:, text_tokens]
    cumulative = torch.cummax(token_probs[:, 0], dim=0)
    scores[:, 0] = cumulative.values
    backpointer[:, 0] = cumulative.indices
    if token_count <= frame_count:
        diagonal = torch.arange(token_count, device=device)
        scores[diagonal, diagonal] = torch.cumsum(
            token_probs[diagonal, diagonal], dim=0
        )
        backpointer[diagonal, diagonal] = diagonal
    else:
        pass
    for frame_index in range(1, frame_count):
        max_token_index = min(frame_index, token_count)
        if max_token_index <= 1:
            continue
        else:
            pass
        token_range = torch.arange(1, max_token_index, device=device)
        skip_scores = scores[frame_index - 1, token_range]
        use_scores = (
            scores[frame_index - 1, token_range - 1]
            + token_probs[frame_index, token_range]
        )
        use_better = use_scores >= skip_scores
        scores[frame_index, token_range] = torch.where(
            use_better, use_scores, skip_scores
        )
        backpointer[frame_index, token_range] = torch.where(
            use_better, frame_index, -1
        )
    positions = torch.zeros(token_count, dtype=torch.long, device=device)
    frame_index = frame_count - 1
    token_index = token_count - 1
    position_index = token_count - 1
    while token_index >= 0:
        if token_index == 0:
            positions[position_index] = backpointer[frame_index, token_index]
            break
        elif backpointer[frame_index, token_index] == -1:
            frame_index -= 1
        else:
            positions[position_index] = backpointer[frame_index, token_index]
            position_index -= 1
            frame_index -= 1
            token_index -= 1
    return positions.tolist()


class TadaAligner(nn.Module):
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = Wav2Vec2ForCTC(wav2vec2_large_config(vocab_size=len(tokenizer)))

    def forward(
        self,
        audio: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_length: torch.Tensor | None = None,
        sample_rate: int = 24000,
    ) -> TadaAlignOutput:
        resampled = torchaudio.functional.resample(audio, sample_rate, 16000)
        resampled = resampled.to(next(self.encoder.parameters()).dtype)
        attention_mask = None
        if audio_length is not None:
            resampled_lengths = (
                audio_length.float() * 16000 / sample_rate
            ).long()
            positions = torch.arange(
                resampled.shape[1], device=audio.device
            ).unsqueeze(0)
            attention_mask = positions < resampled_lengths.unsqueeze(1)
        else:
            pass
        logits = self.encoder(resampled, attention_mask=attention_mask).logits
        if audio_length is None:
            audio_length = torch.tensor(
                [audio.shape[-1]], device=audio.device, dtype=torch.long
            )
        else:
            pass
        input_lengths = (audio_length.float() / sample_rate * 50).ceil().long()
        token_positions, token_masks = self.align_tokens(
            logits, text_tokens, input_lengths
        )
        report = check_alignment(token_positions, int(input_lengths.max().item()))
        return TadaAlignOutput(
            token_positions=token_positions,
            token_masks=token_masks,
            report=report,
            logits=logits,
        )

    def align_tokens(
        self,
        logits: torch.Tensor,
        text_tokens: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eos_id = self.tokenizer.eos_token_id
        all_positions: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        for item_logits, item_tokens in zip(logits, text_tokens):
            valid_tokens = torch.nn.functional.pad(item_tokens, (1, 0))
            masked_logits = torch.full_like(
                item_logits, -float("inf"), device="cpu", dtype=torch.float32
            )
            masked_logits[:, valid_tokens] = item_logits[:, valid_tokens].float().cpu()
            cpu_tokens = item_tokens.cpu()
            selected = align_text_tokens(
                masked_logits, cpu_tokens[cpu_tokens != eos_id]
            )
            position_mask = torch.zeros(
                int(input_lengths.max()), dtype=torch.long, device=logits.device
            )
            position_mask[selected] = 1
            all_positions.append(
                torch.tensor(selected, dtype=torch.long, device=logits.device) + 1
            )
            all_masks.append(position_mask)
        padded_positions = torch.nn.utils.rnn.pad_sequence(
            all_positions, batch_first=True, padding_value=0
        )
        return padded_positions, torch.stack(all_masks, dim=0)


def aligner_subfolder(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"TADA does not ship an aligner for language {language!r}; "
            f"supported languages are {sorted(SUPPORTED_LANGUAGES)}"
        )
    else:
        pass
    if language == "en":
        return "aligner"
    else:
        pass
    return f"aligner-{language}"


__all__ = [
    "TadaAligner",
    "TadaAlignmentReport",
    "TadaAlignOutput",
    "align_text_tokens",
    "aligner_subfolder",
    "check_alignment",
    "wav2vec2_large_config",
]
