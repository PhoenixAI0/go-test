"""Verify the boundary between prepared text examples and served GTP moves."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from go_rl.data_prep import state_record_from_board
from go_rl.gtp_engine import CategoricalMoveChooser, ModelGtpEngine
from go_rl.model_interface import TextGoPolicy
from go_rl.teacher_data import TARGET_RULES
from go_rl.text_distill import prepare_text_record


class CharacterChatTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) + 2 for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token - 2) for token in tokens if token > 1)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "<user>" + messages[0]["content"] + "</user><assistant><think></think>"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        ids = torch.tensor([self.encode(text, add_special_tokens=add_special_tokens)])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


@pytest.mark.parametrize("encoding", ["grid", "compact", "stones"])
def test_gtp_inference_matches_training_prefix_after_capture_and_full_history(encoding):
    tokenizer = CharacterChatTokenizer()
    calls = []

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(max_position_embeddings=4096)

        def generate(self, input_ids, **kwargs):
            calls.append(input_ids[0].tolist())
            allowed = kwargs["prefix_allowed_tokens_fn"]
            # C3 is a ko recapture. It must not become allowed after the board
            # passes through the GTP -> model observation boundary.
            with pytest.raises(ValueError, match="outside"):
                allowed(0, torch.cat((input_ids[0], torch.tensor(tokenizer.encode("C3")))))
            suffix = [*tokenizer.encode("PASS"), tokenizer.eos_token_id]
            for index, token in enumerate(suffix):
                assert token in allowed(0, torch.cat((input_ids[0], torch.tensor(suffix[:index], dtype=torch.long))))
            return torch.cat((input_ids, torch.tensor([suffix])), dim=1)

    policy = TextGoPolicy(Model(), tokenizer, encoding=encoding)
    engine = ModelGtpEngine(lambda board, **kwargs: policy.predict(board, **kwargs).move)
    engine.execute("kata-set-rules", [TARGET_RULES])
    engine.execute("komi", ["6.5"])
    for player, move in [("b", "B3"), ("w", "C3"), ("b", "D3"), ("w", "B2"),
                         ("b", "C4"), ("w", "C1"), ("b", "A19"), ("w", "D2"), ("b", "C2")]:
        engine.execute("play", [player, move])
    assert engine.board.next_player() == "W" and engine.board.current_board()[16][2] == 0
    record = {**state_record_from_board(engine.board), "game_id": "real-history-fixture",
              "rules": TARGET_RULES, "komi": 6.5, "response": "PASS"}
    prepared = prepare_text_record(record, tokenizer, encoding, max_seq_length=4096)
    assert engine.execute("genmove", ["w"]) == "pass"
    assert calls == [prepared["prompt_ids"]]
    assert prepared["answer_ids"] == [*tokenizer.encode("PASS"), tokenizer.eos_token_id]
    prompt = tokenizer.decode(calls[0])
    assert "to_play: W" in prompt and "komi: 6.5" in prompt
    assert "recent_moves: W:C3 B:D3 W:B2 B:C4 W:C1 B:A19 W:D2 B:C2" in prompt


@pytest.mark.parametrize("override", [False, True])
def test_text_gtp_loads_saved_encoding_and_context_defaults_with_explicit_overrides(tmp_path, monkeypatch, override):
    spec = importlib.util.spec_from_file_location(
        "model_gtp_test", Path(__file__).resolve().parents[1] / "scripts" / "model_gtp.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    (tmp_path / "go_text.json").write_text(json.dumps(
        {"encoding": "grid", "board_size": 9, "rules": TARGET_RULES, "komi": 6.5}))
    observations = []

    class Policy:
        def __init__(self, model, tokenizer, **kwargs):
            observations.append(kwargs["encoding"])

        def predict(self, board, **kwargs):
            observations.append((board.size, board.next_player(), kwargs))
            return SimpleNamespace(move="PASS")

    def run(engine):
        observations.append((engine.board.size, engine.komi, engine.rules))
        # Match commands govern the current game, regardless of startup defaults.
        engine.execute("boardsize", ["19"])
        engine.execute("komi", ["5.5"])
        engine.execute("kata-set-rules", ["japanese"])
        engine.execute("genmove", ["b"])

    monkeypatch.setattr("go_rl.model_interface.load_text_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr("go_rl.model_interface.TextGoPolicy", Policy)
    monkeypatch.setattr(runner.ModelGtpEngine, "run", run)
    argv = ["model_gtp.py", "--model", str(tmp_path), "--device", "cpu"]
    if override:
        argv += ["--encoding", "stones", "--board-size", "13", "--rules", "chinese", "--komi", "7.5"]
    monkeypatch.setattr("sys.argv", argv)
    runner.main()
    assert observations[0] == ("stones" if override else "grid")
    assert observations[1] == ((13, 7.5, "chinese") if override else (9, 6.5, TARGET_RULES))
    assert observations[2] == (19, "B", {"komi": 5.5, "rules": "japanese"})


@pytest.mark.parametrize("readout", ["cells", "cell_attention"])
def test_gtp_spatial_checkpoint_rejects_incompatible_observation_before_forward(readout):
    model = SimpleNamespace(readout=readout)
    with pytest.raises(ValueError, match="requires the cells encoding"):
        CategoricalMoveChooser(model, object(), encoding="compact")
