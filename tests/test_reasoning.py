from __future__ import annotations

import logging

from go_rl.reasoning import (
    detect_reasoning_profile,
    extract_final_answer,
    strip_reasoning_for_reward,
)


class _Tokenizer:
    def __init__(self, name_or_path: str = "") -> None:
        self.name_or_path = name_or_path


def test_extract_qwen_reasoning_final_answer():
    final, has_reasoning, malformed = extract_final_answer(
        "<think>analysis</think> D4",
        profile="qwen3",
    )
    assert final == "D4"
    assert has_reasoning is True
    assert malformed is False


def test_extract_qwen_malformed_boundary():
    final, has_reasoning, malformed = extract_final_answer(
        "<think>analysis D4",
        profile="qwen3",
    )
    assert final == ""
    assert has_reasoning is True
    assert malformed is True


def test_extract_gpt_oss_channels():
    completion = [
        {"channel": "analysis", "content": "thinking"},
        {"channel": "final", "content": "Q16"},
    ]
    final, has_reasoning, malformed = extract_final_answer(completion, profile="gpt_oss")
    assert final == "Q16"
    assert has_reasoning is True
    assert malformed is False


def test_extract_glm_content_ignores_reasoning_content():
    completion = {"reasoning_content": "thought", "content": "C3"}
    final, has_reasoning, malformed = extract_final_answer(completion, profile="glm47")
    assert final == "C3"
    assert has_reasoning is True
    assert malformed is False


def test_strip_reasoning_for_reward_malformed_returns_empty():
    output = strip_reasoning_for_reward("<think>x", profile="qwen3")
    assert output == ""


def test_detect_reasoning_profile_auto():
    assert detect_reasoning_profile("Qwen/Qwen3-8B", _Tokenizer(), "auto") == "qwen3"
    assert detect_reasoning_profile("openai/gpt-oss-20b", _Tokenizer(), "auto") == "gpt_oss"
    assert detect_reasoning_profile("zai-org/GLM-4.7-Flash", _Tokenizer(), "auto") == "glm47"


def test_detect_reasoning_profile_does_not_map_generic_openai_models():
    profile = detect_reasoning_profile("openai/gpt-4o-mini", _Tokenizer(), "auto")
    assert profile == "plain"


def test_detect_reasoning_profile_warns_and_defaults_on_ambiguous_markers(caplog):
    caplog.set_level(logging.WARNING)
    profile = detect_reasoning_profile("org/qwen-gpt-oss-mix", _Tokenizer(), "auto")
    assert profile == "gpt_oss"
    assert any("Ambiguous reasoning profile markers" in record.message for record in caplog.records)


def test_detect_reasoning_profile_fallback_plain():
    profile = detect_reasoning_profile("some/unknown-model", _Tokenizer(), "auto")
    assert profile == "plain"


def test_qwen_completion_with_opening_tag_in_prompt():
    assert extract_final_answer("analysis</think> D4", "qwen3") == ("D4", True, False)


def test_thinking_controls_are_not_generation_kwargs():
    from go_rl.config import ReasoningConfig
    from go_rl.reasoning import build_generation_kwargs
    assert build_generation_kwargs(ReasoningConfig(), "qwen3") == {}


def test_qwen_profile_survives_local_adapter_path():
    class Qwen2TokenizerFast:
        name_or_path = "outputs/cuda_run/grpo_9x9"

    assert detect_reasoning_profile("outputs/cuda_run/grpo_9x9", Qwen2TokenizerFast(), "auto") == "qwen3"
