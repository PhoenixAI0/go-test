from __future__ import annotations

from collections import defaultdict

import pytest

torch = pytest.importorskip("torch")

from go_rl.config import ReasoningConfig
from go_rl.reasoning_grpo_trainer import ReasoningGRPOTrainer


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _Model:
    training = True


def _tensor_ids(text: str) -> torch.Tensor:
    return torch.tensor([ord(ch) for ch in text], dtype=torch.long)


def test_reasoning_grpo_trainer_masks_reasoning_tokens(monkeypatch):
    trainer = ReasoningGRPOTrainer.__new__(ReasoningGRPOTrainer)
    trainer.reasoning_config = ReasoningConfig(profile="qwen3", optimize_final_only=True)
    trainer.reasoning_profile = "qwen3"
    trainer.processing_class = _Tokenizer()
    trainer.model = _Model()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    text = "<think>r</think> D4"
    completion_ids = _tensor_ids(text).unsqueeze(0)
    completion_mask = torch.ones_like(completion_ids)

    def fake_super(_self, _inputs):
        return {
            "completion_ids": completion_ids.clone(),
            "completion_mask": completion_mask.clone(),
        }

    monkeypatch.setattr(
        "trl.trainer.grpo_trainer.GRPOTrainer._generate_and_score_completions",
        fake_super,
    )

    output = ReasoningGRPOTrainer._generate_and_score_completions(trainer, inputs=[{}])
    expected = [0] * (len(text) - 2) + [1, 1]
    assert output["completion_mask"].squeeze(0).tolist() == expected
    assert trainer._metrics["train"]["reasoning/masked_token_ratio"]
    assert trainer._metrics["train"]["reasoning/final_span_mean_len"][-1] == 2.0


def test_reasoning_grpo_trainer_disabled_keeps_mask(monkeypatch):
    trainer = ReasoningGRPOTrainer.__new__(ReasoningGRPOTrainer)
    trainer.reasoning_config = ReasoningConfig(enabled=False)
    trainer.reasoning_profile = "plain"
    trainer.processing_class = _Tokenizer()
    trainer.model = _Model()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    completion_ids = _tensor_ids("D4").unsqueeze(0)
    completion_mask = torch.ones_like(completion_ids)

    def fake_super(_self, _inputs):
        return {
            "completion_ids": completion_ids.clone(),
            "completion_mask": completion_mask.clone(),
        }

    monkeypatch.setattr(
        "trl.trainer.grpo_trainer.GRPOTrainer._generate_and_score_completions",
        fake_super,
    )

    output = ReasoningGRPOTrainer._generate_and_score_completions(trainer, inputs=[{}])
    assert output["completion_mask"].equal(completion_mask)
