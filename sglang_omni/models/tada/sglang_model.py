# SPDX-License-Identifier: Apache-2.0
"""SGLang Llama backbone with the TADA acoustic head."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM
from torch import nn
from transformers import LlamaConfig

from sglang_omni.models.tada.components.flow_head import TadaVibeVoiceHead
from sglang_omni.models.weight_loader import default_weight_loader

HEAD_PREFIXES = (
    "acoustic_proj.",
    "time_start_embed.",
    "time_end_embed.",
    "acoustic_mask_emb.",
    "bottleneck_proj.",
    "prediction_head.",
)
SKIPPED_PREFIXES = ("_decoder.",)
REQUIRED_CONFIG_FIELDS = (
    "acoustic_dim",
    "num_time_classes",
    "shift_acoustic",
    "head_layers",
    "head_ffn_ratio",
    "acoustic_mean",
    "acoustic_std",
)


def classify_tada_weight(name: str) -> str:
    if name.startswith("model."):
        return "llama"
    elif name.startswith(HEAD_PREFIXES):
        return "head"
    elif name.startswith(SKIPPED_PREFIXES):
        return "skip"
    else:
        pass
    raise ValueError(f"Unexpected TADA checkpoint weight {name!r}")


class TadaSGLangModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        missing_fields = [
            field for field in REQUIRED_CONFIG_FIELDS if not hasattr(config, field)
        ]
        if missing_fields:
            raise ValueError(
                "TADA checkpoint config is missing fields "
                f"{missing_fields}; refusing to fall back to architecture defaults"
            )
        else:
            pass
        if getattr(config, "diffusion_head_type", "vibevoice") != "vibevoice":
            raise ValueError(
                f"Unsupported TADA diffusion head {config.diffusion_head_type!r}"
            )
        else:
            pass
        self.config = config
        self.llama = LlamaForCausalLM(config, quant_config=quant_config, prefix=prefix)
        self.acoustic_proj = nn.Linear(config.acoustic_dim, config.hidden_size)
        self.time_start_embed = nn.Embedding(
            config.num_time_classes, config.hidden_size
        )
        self.time_end_embed = nn.Embedding(
            config.num_time_classes, config.hidden_size
        )
        self.acoustic_mask_emb = nn.Embedding(2, config.hidden_size)
        head_hidden_size = (
            config.bottleneck_dim
            if config.bottleneck_dim is not None
            else config.hidden_size
        )
        self.bottleneck_proj = (
            nn.Linear(config.hidden_size, config.bottleneck_dim)
            if config.bottleneck_dim is not None
            else nn.Identity()
        )
        self.prediction_head = TadaVibeVoiceHead(
            hidden_size=head_hidden_size,
            head_layers=config.head_layers,
            head_ffn_ratio=config.head_ffn_ratio,
            latent_size=config.acoustic_dim
            + 2 * math.ceil(math.log2(config.num_time_classes)),
        )
        self.compute_text_logits = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llama.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> LogitsProcessorOutput:
        del kwargs
        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        else:
            pass
        hidden_states = self.llama.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            request_count = (
                int(extend_seq_lens.numel())
                if extend_seq_lens is not None
                else 1
            )
        else:
            request_count = int(hidden_states.shape[0])
        if self.compute_text_logits and not forward_batch.forward_mode.is_extend():
            logits = torch.matmul(
                hidden_states.float(), self.llama.lm_head.weight.float().T
            )
        else:
            logits = hidden_states.new_empty((request_count, 1))
        return LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        head_parameters = dict(self.named_parameters())
        loaded: set[str] = set()

        def llama_weight_stream() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                kind = classify_tada_weight(name)
                if kind == "llama":
                    yield name, tensor
                elif kind == "head":
                    parameter = head_parameters[name]
                    loader = getattr(parameter, "weight_loader", default_weight_loader)
                    loader(parameter, tensor)
                    loaded.add(name)
                else:
                    pass

        self.llama.load_weights(llama_weight_stream())
        if self.time_start_embed.num_embeddings != int(self.config.num_time_classes):
            raise ValueError(
                "TADA num_time_classes does not match the checkpoint: "
                f"config={self.config.num_time_classes} "
                f"embedding={self.time_start_embed.num_embeddings}"
            )
        else:
            pass
        if self.acoustic_proj.in_features != int(self.config.acoustic_dim):
            raise ValueError(
                "TADA acoustic_dim does not match the checkpoint: "
                f"config={self.config.acoustic_dim} "
                f"projection={self.acoustic_proj.in_features}"
            )
        else:
            pass
        return loaded


EntryClass = TadaSGLangModel

__all__ = ["TadaSGLangModel", "classify_tada_weight"]
