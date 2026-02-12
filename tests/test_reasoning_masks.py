from __future__ import annotations

from go_rl.reasoning import build_final_only_mask


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_build_mask_qwen_reasoning_keeps_final_only():
    text = "<think>analysis</think> D4"
    mask, malformed, final_len = build_final_only_mask(
        completion=text,
        completion_ids=_ids(text),
        tokenizer=_Tokenizer(),
        profile="qwen3",
        malformed_policy="mask_all",
    )
    expected = [0] * (len(text) - len("D4")) + [1, 1]
    assert mask == expected
    assert malformed is False
    assert final_len == 2


def test_build_mask_malformed_masks_all():
    text = "<think>analysis D4"
    mask, malformed, final_len = build_final_only_mask(
        completion=text,
        completion_ids=_ids(text),
        tokenizer=_Tokenizer(),
        profile="qwen3",
        malformed_policy="mask_all",
    )
    assert mask == [0] * len(text)
    assert malformed is True
    assert final_len == 0


def test_build_mask_plain_profile_keeps_all():
    text = "D4"
    mask, malformed, final_len = build_final_only_mask(
        completion=text,
        completion_ids=_ids(text),
        tokenizer=_Tokenizer(),
        profile="plain",
        malformed_policy="mask_all",
    )
    assert mask == [1, 1]
    assert malformed is False
    assert final_len == 2
