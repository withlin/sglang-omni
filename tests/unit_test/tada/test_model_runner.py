# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.tada.model_runner import (
    next_token_policy,
    should_collect_predicted,
)


def test_teacher_forcing_returns_scheduled_tokens() -> None:
    assert (
        next_token_policy(position=11, schedule_length=20, extra_steps=0)
        == "schedule"
    )
    assert (
        next_token_policy(position=18, schedule_length=20, extra_steps=0)
        == "schedule"
    )


def test_last_transcript_position_samples_only_with_extra_steps() -> None:
    assert (
        next_token_policy(position=19, schedule_length=20, extra_steps=0) == "finish"
    )
    assert (
        next_token_policy(position=19, schedule_length=20, extra_steps=2) == "sample"
    )
    assert (
        next_token_policy(position=20, schedule_length=20, extra_steps=2) == "sample"
    )
    assert (
        next_token_policy(position=21, schedule_length=20, extra_steps=2) == "finish"
    )


def test_frames_belong_to_prompt_until_the_prompt_runs_out() -> None:
    assert (
        should_collect_predicted(position=12, shift_acoustic=5, prompt_frames=10)
        is False
    )
    assert (
        should_collect_predicted(position=15, shift_acoustic=5, prompt_frames=10)
        is True
    )
