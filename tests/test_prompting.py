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


def test_parse_prompt_preserves_fully_occupied_rows():
    board = BoardState.from_matrix(3, [[1, 2, 1], [0, 0, 0], [0, 0, 0]], "W")
    assert parse_prompt(board.to_prompt()).current_board() == board.current_board()


def test_generation_inputs_accepts_batch_encoding():
    from collections import UserDict
    from go_rl.prompting import build_generation_inputs

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return UserDict({"input_ids": [1, 2, 3]})

        def __call__(self, *args, **kwargs):
            raise AssertionError("Chat template output must not be discarded")

    assert build_generation_inputs(Tokenizer(), "move:")["input_ids"] == [1, 2, 3]
