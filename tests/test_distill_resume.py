"""Exercise both real CLI training/resume paths with local tiny Qwen models."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.model_interface import action_to_move, legal_action_mask, move_to_action


def _local_qwen(path: Path) -> None:
    tokens = ["[PAD]", "[EOS]", "[UNK]", *[action_to_move(action) for action in range(362)]]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        eos_token="[EOS]", unk_token="[UNK]",
                                        model_input_names=["input_ids", "attention_mask"])
    tokenizer.chat_template = "{{ messages[0]['content'] }} {% if add_generation_prompt %}move:{% endif %}"
    torch.manual_seed(31)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokens), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, eos_token_id=1, max_position_embeddings=1024,
        attention_dropout=.2,
    ))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def _record(game: str, moves: list[str], response: str, split: str) -> dict:
    board = BoardState.empty(19)
    for move in moves:
        board.apply_move(board.next_player(), move, strict=True)
    legal = legal_action_mask(board)
    target = [0.] * 362
    target[move_to_action(response)] = .8
    target[-1] = .2
    return {**state_record_from_board(board), "game_id": game, "split": split,
            "rules": "koPOSITIONALscoreAREAtaxNONEsui0", "komi": 7.5,
            "response": response, "policy_target": target, "legal_mask": legal,
            "teacher_value": .25}


@pytest.mark.parametrize("kind", ["policy", "text"])
def test_direct_distillation_rejects_final_test_before_loading_model(tmp_path, monkeypatch, kind):
    train_path, eval_path = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train_path.write_text(json.dumps(_record("train", ["D4", "Q16"], "Q4", "train")) + "\n")
    eval_path.write_text(json.dumps(_record("final", ["K10", "J11"], "D4", "test")) + "\n")
    script = Path(__file__).resolve().parents[1] / f"scripts/run_{kind}_distill.py"
    spec = importlib.util.spec_from_file_location(f"{kind}_distill_role_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def no_model_load(*args, **kwargs):
        raise AssertionError("A final-test validation source must fail before model loading")

    monkeypatch.setattr(runner, f"create_{kind}_model", no_model_load)
    monkeypatch.setattr(sys, "argv", [str(script), "--train-data", str(train_path),
                                     "--eval-data", str(eval_path), "--device", "cpu",
                                     "--output-dir", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="Evaluation.*pinned.*test"):
        runner.main()


@pytest.mark.parametrize("kind", ["policy", "text"])
@pytest.mark.parametrize("symmetries", [1, 8])
def test_actual_distillation_cli_resume_matches_uninterrupted_weights(tmp_path, monkeypatch, kind, symmetries):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    base = tmp_path / "base"
    _local_qwen(base)
    training = [
        _record("training-one", ["D4", "Q16"], "Q4", "train"),
        _record("training-two", ["C3", "R17", "Q5"], "D16", "train"),
    ]
    validation = [_record("validation", ["K10", "J11", "L12", "H9"], "D4", "validation")]
    train_path, eval_path = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in training))
    eval_path.write_text("".join(json.dumps(row) + "\n" for row in validation))
    script = Path(__file__).resolve().parents[1] / f"scripts/run_{kind}_distill.py"
    spec = importlib.util.spec_from_file_location(f"{kind}_distill_resume_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Force selection to prefer the pre-update model while still exercising the
    # real forward evaluation. A resume must carry that older best checkpoint,
    # even though its current weights and later metrics are different.
    evaluation_name = "evaluate_nll" if kind == "text" else "evaluate"
    metric_name = "completion_token_nll" if kind == "text" else "policy_loss"
    original_evaluate = getattr(runner, evaluation_name)
    evaluations = 0

    def worsening_selection_metric(*args, **kwargs):
        nonlocal evaluations
        result = original_evaluate(*args, **kwargs)
        result[metric_name] = float(evaluations)
        evaluations += 1
        return result

    monkeypatch.setattr(runner, evaluation_name, worsening_selection_metric)
    arguments = [str(script), "--model", str(base), "--train-data", str(train_path),
                 "--eval-data", str(eval_path), "--device", "cpu", "--steps", "2",
                 "--batch-size", "2", "--gradient-accumulation", "2", "--eval-every", "1",
                 "--checkpoint-every", "1", "--lora-rank", "2", "--lora-alpha", "4",
                 "--learning-rate", "0.001", "--seed", "3407", "--symmetries", str(symmetries)]
    if kind == "text":
        arguments += ["--generation-positions", "1", "--max-new-tokens", "2"]
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", [*arguments, "--output-dir", str(uninterrupted)])
    runner.main()
    run_metadata = json.loads((uninterrupted / "run.json").read_text())
    assert run_metadata["train_positions"] == len(training) * symmetries
    assert run_metadata["prepared_data"]["storage"] == "indexed_jsonl"
    assert (uninterrupted / "prepared_train.jsonl").is_file()
    checkpoint = uninterrupted / "checkpoint-00000001"
    assert (checkpoint / "training_state.pt").is_file()
    contract = torch.load(checkpoint / "training_state.pt", weights_only=True)["contract"]
    assert contract["trainer_version"] == 2 and contract["eval_every"] == 1
    assert contract["train_prepared_sha256"] == run_metadata["prepared_data"]["train_sha256"]
    monkeypatch.setattr(sys, "argv", [*arguments, "--output-dir", str(resumed),
                                     "--resume-from", str(checkpoint)])
    runner.main()
    expected_files = sorted(path.relative_to(uninterrupted / "final")
                            for path in (uninterrupted / "final").rglob("*.safetensors"))
    actual_files = sorted(path.relative_to(resumed / "final")
                          for path in (resumed / "final").rglob("*.safetensors"))
    assert expected_files == actual_files and expected_files
    for relative in expected_files:
        expected = load_file(uninterrupted / "final" / relative)
        actual = load_file(resumed / "final" / relative)
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0,
                                       msg=f"{kind}: {relative}:{name}")
    first_metrics = [json.loads(line) for line in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
    resume_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in resume_metrics] == [1, 2]
    assert first_metrics[-1]["train_loss"] == resume_metrics[-1]["train_loss"]
    assert first_metrics[-1]["gradient_norm"] == resume_metrics[-1]["gradient_norm"]
    metadata_name = "go_text.json" if kind == "text" else "go_policy.json"
    saved_best = json.loads((uninterrupted / "best" / metadata_name).read_text())
    resumed_best = json.loads((resumed / "best" / metadata_name).read_text())
    assert saved_best == resumed_best
    if kind == "text":
        from go_rl.rules import normalize_rules
        assert saved_best["board_size"] == 19 and saved_best["komi"] == 7.5
        assert saved_best["rules"] == normalize_rules(training[0]["rules"])
    assert resumed_best["step"] == 0
    assert resumed_best["validation"][metric_name] == 0
    assert json.loads((resumed / "summary.json").read_text())["best_scope"] == "full_run"
    monkeypatch.setattr(sys, "argv", [*arguments, "--output-dir", str(tmp_path / "changed-cadence"),
                                     "--resume-from", str(checkpoint), "--eval-every", "2"])
    with pytest.raises(ValueError, match="contract changed: eval_every"):
        runner.main()
    # Transformers saves this separately from tokenizer_config.json. Its tokens
    # define the training input, so equal weight hashes alone cannot permit it.
    template = checkpoint / "chat_template.jinja"
    assert template.is_file()
    template.write_text(template.read_text() + " changed-input")
    monkeypatch.setattr(sys, "argv", [*arguments, "--output-dir", str(tmp_path / "changed-template"),
                                     "--resume-from", str(checkpoint)])
    with pytest.raises(ValueError, match="weights differ|prepared_sha256"):
        runner.main()
