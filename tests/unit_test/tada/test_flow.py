# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.tada.components.flow_head import (
    TadaFlowOptions,
    solve_flow_matching,
)


class ConstantVelocityHead(torch.nn.Module):
    def forward(self, noisy_images, timesteps, condition):
        del timesteps
        return torch.ones_like(noisy_images) + 0.1 * condition.mean(
            dim=-1, keepdim=True
        )


def test_zero_negative_cfg_uses_negative_and_positive_branches() -> None:
    options = TadaFlowOptions(
        num_flow_matching_steps=1,
        acoustic_cfg_scale=2.0,
        duration_cfg_scale=1.0,
        cfg_schedule="constant",
        time_schedule="uniform",
    )
    output = solve_flow_matching(
        prediction_head=ConstantVelocityHead(),
        speech=torch.zeros(2, 8),
        conditioning=torch.ones(2, 8),
        negative_condition=torch.zeros(2, 8),
        acoustic_dim=6,
        options=options,
    )
    assert torch.allclose(output[:, :6], torch.full((2, 6), 1.2))
    assert torch.allclose(output[:, 6:], torch.full((2, 2), 1.1))


def test_no_cfg_runs_a_single_branch() -> None:
    options = TadaFlowOptions(
        num_flow_matching_steps=1,
        acoustic_cfg_scale=1.0,
        duration_cfg_scale=1.0,
        cfg_schedule="constant",
        time_schedule="uniform",
    )
    output = solve_flow_matching(
        prediction_head=ConstantVelocityHead(),
        speech=torch.zeros(1, 4),
        conditioning=torch.ones(1, 4),
        negative_condition=torch.zeros(1, 4),
        acoustic_dim=3,
        options=options,
    )
    assert torch.allclose(output, torch.full((1, 4), 1.1))
