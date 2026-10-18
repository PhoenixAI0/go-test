"""Exercise real TRL/PEFT training on a tiny local CPU model (no downloads).

The real shared loader is used; only the external KataGo engine is substituted.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,config_name", [
    ("run_sft", "sft_warm_start.yaml"),
    ("run_grpo", "grpo_9x9.yaml"),
    ("run_grpo_fullgame", "grpo_fullgame_19x19.yaml"),
])
def test_training_entrypoint_real_trainer(tmp_path, monkeypatch, script, config_name):
    import torch
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from go_rl.board import BoardState
    from go_rl.katago import KataGoResult

    # The shared loader selects its device independently of Trainer.use_cpu.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "A1": 3, "PASS": 4}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"])
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=5, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=1, eos_token_id=2))

    base_dir = tmp_path / "base"
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    model_path = base_dir
    if script != "run_sft":
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2,
            target_modules="all-linear", task_type="CAUSAL_LM"))
        # Store the base path exactly as the real loader does.
        model.peft_config["default"].base_model_name_or_path = str(base_dir)
        model_path = tmp_path / "previous_stage"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "default_num_workers", lambda: 0)
    monkeypatch.setattr(module, "precision_kwargs", lambda *args: {"use_cpu": True, "bf16": False, "fp16": False})
    monkeypatch.setattr(module, "configure_torch_for_speed", lambda **kwargs: None)
    monkeypatch.setattr(module, "get_optimal_attention_backend", lambda: "sdpa")
    monkeypatch.setattr(module, "write_run_log", lambda *args, **kwargs: tmp_path / "log.json")
    if script != "run_sft":
        engine = SimpleNamespace(start=lambda: None, close=lambda: None,
            analyze_position=lambda board: KataGoResult(0.5, {}, score_lead=0, best_move="PASS"),
            analyze_positions=lambda boards: [KataGoResult(0.5, {}, score_lead=0) for _ in boards],
            choose_move=lambda *args, **kwargs: "PASS")
        monkeypatch.setattr(module, "build_katago_service", lambda cfg: engine)
    config = yaml.safe_load((ROOT / "config" / config_name).read_text())
    training = config if script == "run_sft" else config["training"]
    training.update(model_name=str(model_path), output_dir=str(tmp_path / "model"),
                    log_dir=str(tmp_path / "logs"), max_seq_length=128, quantization_bits=0,
                    use_8bit_optimizer=False, batch_size=1, lora_rank=2, lora_alpha=2)
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({"prompt": BoardState.empty(3).to_prompt(), "response": "A1"}) + "\n" for _ in range(4)))
    if script == "run_sft":
        config.update(dataset_path=str(data), eval_split_ratio=0, use_packing=False)
    else:
        config["dataset"] = str(data)
        training["gradient_accumulation"] = 2
        config["rollout"].update(num_samples=2, max_tokens=4)
        config["katago"].update(binary_path="/mock/katago", model_path="/mock/model")
        if script == "run_grpo_fullgame":
            config["full_game"].update(max_plies=2, max_model_turns=1)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", [script, "--config", str(cfg_path), "--smoke", "--gradient-checkpointing"])
    module.main()
    assert (tmp_path / "model/adapter_config.json").is_file()
    assert (tmp_path / "model/adapter_model.safetensors").is_file()
    assert (tmp_path / "model/tokenizer.json").is_file()

    # Reload the saved stage with the production loader and verify exact weights.
    from go_rl.training_utils import load_training_model
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    from go_rl.config import SftConfig, TrainingConfig
    stage_config = training.copy()
    stage_config["model_name"] = str(tmp_path / "model")
    cls = SftConfig if script == "run_sft" else TrainingConfig
    continued, _ = load_training_model(cls(**stage_config), gradient_checkpointing=True)
    expected = load_file(str(tmp_path / "model/adapter_model.safetensors"))
    actual = get_peft_model_state_dict(continued)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert all(parameter.requires_grad == ("lora_" in name)
               for name, parameter in continued.named_parameters())
