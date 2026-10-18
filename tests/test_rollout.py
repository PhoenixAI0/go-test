import pytest

from go_rl.rollout import parse_move_answer


@pytest.mark.parametrize("answer,expected", [
    ("D4", "D4"),
    ("Q16", "Q16"),
    ("T19", "T19"),
    ("PASS", "PASS"),
    (" D4\n", "D4"),  # Whitespace around the answer is not a format error.
])
def test_parse_move_answer_accepts_bare_moves(answer, expected):
    assert parse_move_answer(answer) == expected


@pytest.mark.parametrize("answer", [
    "", "   ", "d4", "pass", "D4.", "D 4", "D04", "D0", "D20", "I4", "Move: D4",
    "I'll play D4", "D4 D5", "D4\nQ16", "```D4```", "I will PASS", "PASS please",
])
def test_parse_move_answer_rejects_other_formats(answer):
    assert parse_move_answer(answer) == ""
