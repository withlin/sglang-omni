# SPDX-License-Identifier: Apache-2.0
"""Local self-attention blocks shared by the TADA encoder and decoder."""

import math
from typing import Literal

import torch
from torch import nn


def create_encoder_attention_mask(
    text_token_mask: torch.Tensor,
    version: Literal["v1", "v2"] = "v2",
) -> torch.Tensor:
    """Encoder block mask where True means masked (cannot attend).

    Marked positions start blocks and hide themselves from unmarked positions.
    """
    block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
    block_ids_i = block_ids.unsqueeze(2)
    block_ids_j = block_ids.unsqueeze(1)
    if version == "v1":
        same_block = block_ids_i == block_ids_j
        block_ids_j_excluding_last = torch.where(
            text_token_mask.bool(), -10, block_ids_j[:, 0, :]
        ).unsqueeze(1)
        next_block = block_ids_j_excluding_last == (block_ids_i + 1)
        return ~(same_block | next_block)
    elif version == "v2":
        # Blocks are the segments between consecutive marked positions.
        block_ids = torch.cumsum(text_token_mask, dim=1)
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_i == block_ids_j
        is_marked_i = text_token_mask.unsqueeze(2).bool()
        is_marked_j = text_token_mask.unsqueeze(1).bool()
        same_block_valid = same_block & (~is_marked_j | (is_marked_i & same_block))
        previous_block = block_ids_j == (block_ids_i - 1)
        previous_block_valid = previous_block & ~is_marked_j
        can_attend = same_block_valid | (is_marked_i & previous_block_valid)
        return ~can_attend
    else:
        pass
    raise ValueError(f"Unknown encoder attention version: {version}")


def create_decoder_attention_mask(
    text_token_mask: torch.Tensor,
    version: Literal["v1", "v2"] = "v2",
) -> torch.Tensor:
    """Decoder block mask where True means masked (cannot attend).

    Marked positions belong to the preceding block; v2 allows same-block and
    previous-block attention and v1 adds the causal unmarked rule.
    """
    block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
    block_ids_i = block_ids.unsqueeze(2)
    block_ids_j = block_ids.unsqueeze(1)
    if version == "v2":
        can_attend = (block_ids_j == block_ids_i) | (block_ids_j == (block_ids_i - 1))
        return ~can_attend
    elif version == "v1":
        same_block = block_ids_i == block_ids_j
        sequence_length = text_token_mask.shape[1]
        positions = torch.arange(sequence_length, device=text_token_mask.device)
        position_i = positions.unsqueeze(1)
        position_j = positions.unsqueeze(0)
        is_marked_i = text_token_mask.unsqueeze(2).bool()
        marked_causal = (position_j <= position_i).unsqueeze(0) & is_marked_i
        past = (position_j < position_i).unsqueeze(0)
        current_block_forward = (position_j >= position_i) & same_block
        non_marked_attention = (past | current_block_forward) & ~is_marked_i
        return ~(marked_causal | non_marked_attention)
    else:
        pass
    raise ValueError(f"Unknown decoder attention version: {version}")


class LocalSelfAttention(nn.Module):
    """Local self-attention with RoPE and checkpoint-compatible buffers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "rope_freqs", self.compute_rope_freqs(self.head_dim, max_seq_len)
        )
        self.max_seq_len = max_seq_len
        self.register_buffer(
            "_precomputed_mask", self.create_full_mask(max_seq_len)
        )

    @staticmethod
    def create_full_mask(max_seq_len: int) -> torch.Tensor:
        positions = torch.arange(max_seq_len).unsqueeze(0)
        positions_t = positions.transpose(0, 1)
        distances = positions - positions_t
        return distances.abs() >= 0

    @staticmethod
    def compute_rope_freqs(head_dim: int, max_seq_len: int) -> torch.Tensor:
        inverse_frequencies = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, inverse_frequencies)
        return torch.stack([frequencies.cos(), frequencies.sin()], dim=-1)

    def apply_rope(
        self, hidden_states: torch.Tensor, sequence_length: int
    ) -> torch.Tensor:
        batch_size, num_heads, _, head_dim = hidden_states.shape
        frequencies = self.rope_freqs[:sequence_length]
        cosines = frequencies[..., 0]
        sines = frequencies[..., 1]
        paired = hidden_states.reshape(
            batch_size, num_heads, sequence_length, head_dim // 2, 2
        )
        first = paired[..., 0]
        second = paired[..., 1]
        rotated_first = (
            first * cosines.unsqueeze(0).unsqueeze(0)
            - second * sines.unsqueeze(0).unsqueeze(0)
        )
        rotated_second = (
            first * sines.unsqueeze(0).unsqueeze(0)
            + second * cosines.unsqueeze(0).unsqueeze(0)
        )
        rotated = torch.stack([rotated_first, rotated_second], dim=-1)
        return rotated.reshape(batch_size, num_heads, sequence_length, head_dim)

    def create_local_mask(self, sequence_length: int) -> torch.Tensor:
        return self._precomputed_mask[:sequence_length, :sequence_length]  # noqa: leading-underscore  # checkpoint buffer name

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.apply_rope(query, sequence_length)
        key = self.apply_rope(key, sequence_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        if mask is None:
            local_mask = self.create_local_mask(sequence_length)
            attention_scores = attention_scores.masked_fill(
                local_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        elif mask.dim() == 2:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        elif mask.dim() == 3:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(1), float("-inf")
            )
        else:
            raise ValueError(f"Mask should have 2 or 3 dimensions, got {mask.dim()}")
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, d_model
        )
        output = self.dropout(self.out_proj(attention_output))
        return self.layer_norm(hidden_states + output)


class LocalAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 8192,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        else:
            pass
        self.self_attn = LocalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            causal=causal,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.self_attn(hidden_states, mask=mask)
        return self.norm(hidden_states + self.ffn(hidden_states))


class LocalAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_input: int | None = None,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 8192,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LocalAttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    max_seq_len=max_seq_len,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        if d_input is not None and d_input != d_model:
            self.input_proj: nn.Module = nn.Linear(d_input, d_model)
        else:
            self.input_proj = nn.Identity()

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return self.final_norm(hidden_states)


__all__ = [
    "LocalAttentionEncoder",
    "LocalAttentionEncoderLayer",
    "LocalSelfAttention",
    "create_decoder_attention_mask",
    "create_encoder_attention_mask",
]
