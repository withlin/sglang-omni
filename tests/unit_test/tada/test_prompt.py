# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.tada.prompt import (
    TadaSpecialTokens,
    build_prompt_layout,
    build_text_schedule,
)

SPECIALS = TadaSpecialTokens(
    bos_id=1,
    eos_id=2,
    eot_id=3,
    pad_id=4,
    start_header_id=5,
    end_header_id=6,
)


class StubTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.next_id = 100

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for _ in text.split():
            ids.append(self.next_id)
            self.next_id += 1
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|eot_id|>": 3,
            "<|finetune_right_pad_id|>": 4,
            "<|start_header_id|>": 5,
            "<|end_header_id|>": 6,
        }[token]


def test_text_schedule_puts_prefix_after_bos_and_five_eot_at_the_end() -> None:
    tokenizer = StubTokenizer()
    schedule = build_text_schedule(
        tokenizer=tokenizer,
        reference_text="a b",
        target_text="c d e",
        shift_acoustic=5,
        system_prompt="sys",
        user_turn_prompt=None,
    )
    header_ids = tokenizer.encode(
        "<|start_header_id|>system<|end_header_id|>sys<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>",
        add_special_tokens=False,
    )
    assert schedule.input_ids.shape == (1, 1 + len(header_ids) + 2 + 3 + 5)
    assert schedule.input_ids[0, 0].item() == SPECIALS.bos_id
    assert schedule.input_ids[0, -5:].tolist() == [SPECIALS.eot_id] * 5
    assert schedule.reference_token_count == 2
    assert schedule.target_token_count == 3


def make_layout(use_text_in_prompt: bool):
    return build_prompt_layout(
        specials=SPECIALS,
        shift_acoustic=5,
        num_transition_steps=5,
        num_time_classes=256,
        acoustic_dim=4,
        input_ids=torch.tensor(
            [
                [
                    1,
                    5,
                    6,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    200,
                    201,
                    3,
                    3,
                    3,
                    3,
                    3,
                ]
            ]
        ),
        reference_token_positions=torch.tensor(
            [[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
        ),
        reference_token_values=torch.ones(10, 4),
        reference_audio_length=torch.tensor([4800]),
        header_length=2,
        use_text_in_prompt=use_text_in_prompt,
    )


def test_prompt_layout_pads_then_trims_and_masks_text_tokens() -> None:
    layout = make_layout(use_text_in_prompt=False)
    assert layout.acoustic.shape == (1, 7, 4)
    assert layout.acoustic_mask.tolist() == [[1] * 7]
    assert layout.time_before.shape == (1, 7)
    assert layout.time_after.shape == (1, 7)
    assert layout.output_start == 11
    assert layout.prefill_len == 11
    assert layout.input_ids[0, 0].item() == SPECIALS.bos_id
    assert layout.input_ids[0, 1].item() == SPECIALS.start_header_id
    assert layout.input_ids[0, 3].item() == SPECIALS.pad_id
    assert layout.input_ids[0, 6].item() == SPECIALS.pad_id
    assert layout.input_ids[0, 13].item() == 200


def test_prompt_layout_keeps_text_when_requested() -> None:
    layout = make_layout(use_text_in_prompt=True)
    assert layout.input_ids[0, 3].item() == 100
    assert layout.input_ids[0, 6].item() == 103
