from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token = ""
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, return_dict=False, **kwargs):
        text = ""
        if add_generation_prompt:
            text = messages[0]["content"]
        else:
            text = messages[0]["content"] + messages[1]["content"]
        ids = [ord(ch) for ch in text]
        if return_dict:
            return {"input_ids": ids}
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


PROMPT = "\n".join(
    [
        "board_size: 9",
        "to_play: B",
        "board:",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        "move:",
    ]
)


def test_sft_reasoning_completion_mask_masks_thinking_tokens():
    module = _load_script_module("run_sft.py", "run_sft_reasoning_mask_module")
    tokenizer = _Tokenizer()
    sample = module._convert_to_messages(
        {"prompt": PROMPT, "response": "<think>analysis</think> D4"},
    )

    processed = module._add_reasoning_completion_mask(
        sample,
        tokenizer=tokenizer,
        profile="qwen3",
        malformed_policy="mask_all",
        reasoning_enabled=True,
        optimize_final_only=True,
        chat_template_kwargs={},
    )
    prompt_len = len(PROMPT)
    completion_mask = processed["completion_mask"][prompt_len:]
    assert completion_mask[-2:] == [1, 1]
    assert all(v == 0 for v in completion_mask[:-2])


def test_sft_reasoning_disabled_keeps_full_completion_mask():
    module = _load_script_module("run_sft.py", "run_sft_reasoning_mask_disabled_module")
    tokenizer = _Tokenizer()
    sample = module._convert_to_messages(
        {"prompt": PROMPT, "response": "<think>analysis</think> D4"},
    )

    processed = module._add_reasoning_completion_mask(
        sample,
        tokenizer=tokenizer,
        profile="qwen3",
        malformed_policy="mask_all",
        reasoning_enabled=False,
        optimize_final_only=True,
        chat_template_kwargs={},
    )
    prompt_len = len(PROMPT)
    completion_mask = processed["completion_mask"][prompt_len:]
    assert all(v == 1 for v in completion_mask)


def test_sft_generation_prefix_and_single_eos_are_preserved():
    module = _load_script_module("run_sft.py", "sft_prefix_regression")

    class PrefixTokenizer(_Tokenizer):
        eos_token_id = 0
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"]
            return self.encode(messages[0]["content"] + "<think></think>")

    tokenizer = PrefixTokenizer()
    sample = module._convert_to_messages({"prompt": "board", "response": "D4"})
    output = module._add_reasoning_completion_mask(sample, tokenizer, "qwen3", "mask_all", True, True, {})
    boundary = len("board<think></think>")
    assert output["input_ids"] == tokenizer.encode("board<think></think>D4") + [0]
    assert output["completion_mask"] == [0] * boundary + [1, 1, 1]
