"""Training rejects ended area states and historically invalid target-rule games."""
import importlib.util
from pathlib import Path

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.teacher_data import TARGET_RULES
from go_rl.text_distill import prepare_text_record

spec = importlib.util.spec_from_file_location(
    "policy_distill_history_test", Path(__file__).resolve().parents[1] / "scripts" / "run_policy_distill.py")
policy_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_runner)


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(char) + 2 for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token - 2) for token in tokens if token > 1)

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        text = messages[0]["content"]
        return self.encode(text) if tokenize else text


@pytest.fixture(params=["text", "categorical"])
def prepare(request):
    def call(board, rules=TARGET_RULES):
        record = {**state_record_from_board(board), "game_id": "history-fixture",
                  "rules": rules, "komi": 7.5, "response": "PASS"}
        if request.param == "text":
            return prepare_text_record(record, Tokenizer(), max_seq_length=4096)
        return policy_runner.prepare_records([record], Tokenizer(), "compact", "move", 4096, board.size)[0]
    return call


@pytest.mark.parametrize("continued", [False, True])
def test_area_training_rejects_two_pass_boundary_even_after_further_moves(prepare, continued):
    board = BoardState.empty(3)
    for move in ["PASS", "PASS", *(["A1"] if continued else [])]:
        board.apply_move(board.next_player(), move, strict=True)
    assert "PASS" in board.legal_moves(superko=True)  # Current-action legality is insufficient.
    with pytest.raises(ValueError, match="precede two-pass termination; filter or regenerate"):
        prepare(board)


@pytest.mark.parametrize("continued", [False, True])
def test_territory_phase_semantics_are_not_replaced_with_area_termination(prepare, continued):
    board = BoardState.empty(3)
    for move in ["PASS", "PASS", *(["A1"] if continued else [])]:
        board.apply_move(board.next_player(), move, strict=True)
    assert prepare(board, "japanese")


def test_single_pass_remains_a_valid_training_decision(prepare):
    board = BoardState.empty(3)
    board.apply_move("B", "PASS", strict=True)
    assert prepare(board)


def test_training_checks_earlier_superko_even_when_current_response_is_legal(prepare):
    # Three separated copies of the existing C2/C3 ko shape. Alternate the
    # middle one's colors so six alternating captures restore the initial board
    # without an immediate ko recapture or any pass/terminal boundary.
    matrix = [[0] * 19 for _ in range(19)]
    shape = [(16, 1, 1), (16, 3, 1), (15, 2, 1), (16, 2, 2),
             (17, 1, 2), (18, 2, 2), (17, 3, 2)]
    for offset, invert in [(0, False), (6, True), (12, False)]:
        for row, column, color in shape:
            matrix[row][column + offset] = 3 - color if invert else color
    board = BoardState.from_matrix(19, matrix, "B")
    for move in ["C2", "J2", "P2", "C3", "J3", "P3"]:
        board.apply_move(board.next_player(), move, strict=True)
    assert board.current_board() == matrix and not board.has_two_pass_history()
    assert "PASS" in board.legal_moves(superko=True)
    with pytest.raises(ValueError, match="history violates target positional superko"):
        prepare(board)
    assert prepare(board, "chinese")  # Simple ko permits this longer repetition.
