import pytest

from go_rl.board import BoardState
from go_rl.prompting import build_prompt, parse_prompt


def test_build_and_parse_prompt_roundtrip():
    board = BoardState.empty(5)
    board.apply_move("B", "A1")
    board.apply_move("W", "PASS")
    prompt = build_prompt(board)
    parsed = parse_prompt(prompt)
    assert parsed.size == 5
    assert parsed.moves == board.moves


def test_parse_prompt_requires_header():
    with pytest.raises(ValueError):
        parse_prompt("B: A1\nmove:")
