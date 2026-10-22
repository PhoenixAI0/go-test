import pytest

from go_rl.board import BoardState
from go_rl.model_interface import (
    MoveTokenTrie, action_to_move, legal_action_mask, move_to_action, render_position,
)


class CharacterTokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)


@pytest.mark.parametrize("size", [9, 19])
def test_every_action_round_trips(size):
    for action in range(size * size + 1):
        assert move_to_action(action_to_move(action, size), size) == action
    assert action_to_move(0, size) == f"A{size}"
    assert action_to_move(size * size - 1, size) == ("J1" if size == 9 else "T1")
    assert action_to_move(size * size, size) == "PASS"


def test_trie_handles_terminal_prefixes_and_multidigit_rows():
    tokenizer = CharacterTokenizer()
    trie = MoveTokenTrie(tokenizer, ["A1", "A19", "T19", "PASS"])
    assert trie.allowed(tokenizer.encode("A1")) == sorted([ord("9"), 999])
    assert trie.allowed(tokenizer.encode("T19")) == [999]
    for move in ["A1", "A19", "T19", "PASS"]:
        tokens = [*tokenizer.encode(move), 999]
        for i, token in enumerate(tokens):
            assert token in trie.allowed(tokens[:i])
    with pytest.raises(ValueError, match="outside"):
        trie.allowed(tokenizer.encode("B"))


def test_legality_mask_excludes_suicide_and_occupied_points():
    board = BoardState.from_matrix(3, [[0, 2, 0], [2, 0, 2], [0, 2, 0]], "B")
    mask = legal_action_mask(board)
    assert not mask[move_to_action("B2", 3)]
    assert not mask[move_to_action("B3", 3)]
    assert mask[9]


def test_interfaces_keep_orientation_player_and_komi():
    board = BoardState.from_matrix(3, [[1, 0, 0], [0, 0, 0], [0, 0, 2]], "W")
    grid = render_position(board, "grid", komi=6.5)
    assert " 3: X . ." in grid and " 1: . . O" in grid
    assert "to_play: W" in grid and "komi: 6.5" in grid
    assert "B: A3" in render_position(board, "stones")
    assert "W: C1" in render_position(board, "stones")
    assert " 3: X.." in render_position(board, "compact")
    assert render_position(board, "legacy") == board.to_prompt()


def test_actions_reject_out_of_range():
    for action in (-1, 362):
        with pytest.raises(ValueError):
            action_to_move(action)
