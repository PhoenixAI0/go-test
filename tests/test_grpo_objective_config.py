from dataclasses import asdict

import pytest

from go_rl.config import GRPOObjectiveConfig, RolloutConfig


@pytest.mark.parametrize("kwargs", [
    {"loss_type": "invented"}, {"scale_rewards": "invalid"},
    {"importance_sampling_level": "game"}, {"beta": -1},
    {"beta": float("nan")}, {"epsilon": 1}, {"epsilon_high": -0.1},
    {"num_iterations": 0}, {"top_entropy_quantile": 0},
])
def test_invalid_objective_settings_fail_before_loading_model(kwargs):
    with pytest.raises(ValueError):
        GRPOObjectiveConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"temperature": float("nan")},
    {"top_p": 0}, {"top_p": 1.1}, {"top_k": -1},
    {"max_tokens": 0}, {"num_samples": 0},
])
def test_invalid_generation_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        RolloutConfig(**kwargs)


def test_objective_and_sampling_reach_actual_trl_trainer(tmp_path):
    pytest.importorskip("torch")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig
    from go_rl.reasoning_grpo_trainer import ReasoningGRPOTrainer

    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "<eos>": 2,
                                  "go": 3, "D4": 4, "PASS": 5}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                       pad_token="<pad>", eos_token="<eos>")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=6, n_embd=8, n_layer=1, n_head=1, n_positions=32,
        bos_token_id=2, eos_token_id=2, pad_token_id=1,
    ))
    objective = GRPOObjectiveConfig()
    rollout = RolloutConfig(num_samples=2, top_p=1.0, top_k=0)
    args = GRPOConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        report_to=[], per_device_train_batch_size=2, num_generations=2,
        max_prompt_length=8, max_completion_length=4, gradient_checkpointing=False,
        temperature=rollout.temperature, top_p=rollout.top_p, top_k=rollout.top_k,
        **asdict(objective),
    )
    trainer = ReasoningGRPOTrainer(
        model=model, processing_class=tokenizer, args=args,
        train_dataset=Dataset.from_dict({"prompt": ["go", "go"]}),
        reward_funcs=lambda completions, **_: [0.0] * len(completions),
    )
    assert trainer.generation_config.top_p == 1.0
    assert trainer.generation_config.top_k == 0
    assert trainer.loss_type == "dr_grpo"
    assert trainer.scale_rewards == "none"
    assert trainer.epsilon_high == 0.28
    assert trainer.beta == 0.0
    assert not trainer.reasoning_config.optimize_final_only
