# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_tada_pipeline_is_discoverable_by_architecture() -> None:
    config = PIPELINE_CONFIG_REGISTRY.get_config("TadaForCausalLM")
    assert config.__name__ == "TadaPipelineConfig"
