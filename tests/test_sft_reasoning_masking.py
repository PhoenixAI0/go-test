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


def test_sft_reasoning_completion_mask_masks_thinking_tokens():
    module = _load_script_module("run_sft.py", "run_sft_reasoning_mask_module")
    tokenizer = _Tokenizer()
    sample = module._convert_to_messages(
        {"prompt": "board_size: 9\nmove:", "response": "<think>analysis</think> D4"},
        eos_token="",
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
    prompt_len = len("board_size: 9\nmove:")
    completion_mask = processed["completion_mask"][prompt_len:]
    assert completion_mask[-2:] == [1, 1]
    assert all(v == 0 for v in completion_mask[:-2])


def test_sft_reasoning_disabled_keeps_full_completion_mask():
    module = _load_script_module("run_sft.py", "run_sft_reasoning_mask_disabled_module")
    tokenizer = _Tokenizer()
    sample = module._convert_to_messages(
        {"prompt": "board_size: 9\nmove:", "response": "<think>analysis</think> D4"},
        eos_token="",
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
    prompt_len = len("board_size: 9\nmove:")
    completion_mask = processed["completion_mask"][prompt_len:]
    assert all(v == 1 for v in completion_mask)
