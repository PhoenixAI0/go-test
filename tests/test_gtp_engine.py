from __future__ import annotations

import io
import json
import shlex
import sys

import pytest

from go_rl.gtp_engine import CategoricalMoveChooser, ModelGtpEngine
from go_rl.gtp_match import GtpEngine


def test_gtp_request_ids_comments_multiline_and_stdout_isolation():
    def chooser(board, **kwargs):
        print("model diagnostic")
        assert board.size == 19 and kwargs["komi"] == 7.5
        return "T19"

    engine = ModelGtpEngine(chooser)
    output, errors = io.StringIO(), io.StringIO()
    engine.run(io.StringIO("# comment\n\n1 protocol_version\n2 genmove b # move\n3 showboard\n4 quit\n5 name\n"), output, errors)
    text = output.getvalue()
    assert text.startswith("=1 2\n\n=2 t19\n\n=3 19 ")
    assert text.endswith("=4\n\n")
    assert "model diagnostic" not in text and "model diagnostic" in errors.getvalue()
    assert "=5" not in text
    assert engine.board.current_board()[0][18] == 1


def test_invalid_commands_do_not_mutate_state_or_claim_scoring():
    engine = ModelGtpEngine(lambda *_args, **_kwargs: "PASS")
    engine.execute("play", ["black", "D4"])
    state = engine.board.current_board()
    for command, args in [("play", ["w", "D4"]), ("play", ["x", "A1"]),
                          ("boardsize", ["20"]), ("komi", ["nan"]),
                          ("komi", ["6.2"]), ("play", ["w"]),
                          ("kata-set-rules", ["aga"]), ("final_score", [])]:
        with pytest.raises(ValueError):
            engine.execute(command, args)
        assert engine.board.current_board() == state
        assert engine.komi == 7.5 and engine.rules == "chinese"
    assert engine.execute("known_command", ["final_score"]) == "false"


def test_illegal_or_failed_model_output_leaves_state_unchanged():
    engine = ModelGtpEngine(lambda *_args, **_kwargs: "D4")
    engine.execute("play", ["b", "D4"])
    before = engine.board.current_board()
    with pytest.raises(ValueError, match="illegal move"):
        engine.execute("genmove", ["w"])
    assert engine.board.current_board() == before

    def failing_policy(board, **kwargs):
        board.apply_move("W", "A1")  # Even a misbehaving callback gets its own board.
        raise RuntimeError("model broke")

    engine.choose_move = failing_policy
    output, errors = io.StringIO(), io.StringIO()
    engine.run(io.StringIO("7 genmove w\n8 quit\n"), output, errors)
    assert output.getvalue() == "?7 model error; see stderr\n\n=8\n\n"
    assert "model broke" in errors.getvalue()
    assert engine.board.current_board() == before


def test_undo_restores_capture_and_ko_context():
    engine = ModelGtpEngine(lambda *_args, **_kwargs: "PASS", board_size=9)
    for color, move in [("b", "B3"), ("b", "D3"), ("b", "C4"),
                        ("w", "C3"), ("w", "B2"), ("w", "C1"), ("w", "D2")]:
        engine.execute("play", [color, move])
    before_capture = engine.board.current_board()
    engine.execute("play", ["b", "C2"])
    after_capture = engine.board.current_board()
    engine.execute("play", ["w", "PASS"])
    engine.execute("undo", [])
    assert engine.board.current_board() == after_capture
    with pytest.raises(ValueError, match="illegal move"):
        engine.execute("play", ["w", "C3"])
    engine.execute("undo", [])
    assert engine.board.current_board() == before_capture
    assert engine.board.next_player() == "B"


def test_rules_and_komi_are_forwarded_and_unsupported_sizes_fail():
    calls = []

    def choose(board, **kwargs):
        calls.append((board.next_player(), kwargs))
        return "PASS"

    engine = ModelGtpEngine(choose, supported_sizes={19})
    engine.execute("komi", ["6.5"])
    engine.execute("kata-set-rules", ["koPOSITIONALscoreAREAtaxNONEsui0"])
    assert engine.superko
    assert json.loads(engine.execute("kata-get-rules", []))["ko"] == "POSITIONAL"
    engine.execute("genmove", ["white"])
    assert calls == [("W", {"komi": 6.5, "rules": "koPOSITIONALscoreAREAtaxNONEsui0"})]
    assert engine.board.moves == [("W", None)]
    with pytest.raises(ValueError, match="unacceptable size"):
        engine.execute("boardsize", ["9"])
    assert engine.board.size == 19
    engine.execute("clear_board", [])
    assert not engine.board.moves
    assert engine.komi == 6.5


def test_model_engine_real_subprocess_protocol():
    script = (
        "from go_rl.gtp_engine import ModelGtpEngine; "
        "ModelGtpEngine(lambda board, **kwargs: board.legal_moves()[0]).run()"
    )
    engine = GtpEngine("Model", shlex.join([sys.executable, "-c", script]))
    try:
        assert engine.send("protocol_version") == "2"
        engine.send("boardsize 19")
        engine.send("komi 7.5")
        assert engine.send("genmove b") == "a19"
        engine.send("play w T1")
        board = engine.send("showboard")
        assert board.startswith("19 X") and " 1 ." in board
        engine.send("undo")
        assert engine.send("genmove w") == "b19"
    finally:
        engine.close()


def test_categorical_adapter_uses_checkpoint_inputs_and_legal_mask(monkeypatch):
    import torch
    from types import SimpleNamespace

    calls = []

    class Model:
        board_size = 2
        device = "cpu"
        backbone = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))

        def eval(self):
            return self

        def __call__(self, **kwargs):
            calls.append(kwargs)
            logits = torch.tensor([[20.0, 10.0, 2.0, 1.0, 0.0]])
            if kwargs["legal_mask"] is not None:
                logits = logits.masked_fill(~kwargs["legal_mask"], float("-inf"))
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr("go_rl.policy_model.encode_policy_position", lambda *args, **kwargs: [1, 2])
    tokenizer = SimpleNamespace(pad_token_id=0)
    chooser = CategoricalMoveChooser(Model(), tokenizer)
    engine = ModelGtpEngine(chooser, board_size=2)
    engine.execute("play", ["b", "A2"])
    assert engine.execute("genmove", ["w"]) == "b2"
    assert calls[0]["legal_mask"].tolist() == [[False, True, True, True, True]]
    assert calls[0]["input_ids"].tolist() == [[1, 2]]
