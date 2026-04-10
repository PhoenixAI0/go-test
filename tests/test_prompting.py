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
    assert parsed.next_player() == board.next_player()
    assert parsed.current_board() == board.current_board()


def test_parse_prompt_requires_header():
    with pytest.raises(ValueError):
        parse_prompt("to_play: B\nboard:\n.....\n.....\n.....\n.....\n.....\nmove:")
