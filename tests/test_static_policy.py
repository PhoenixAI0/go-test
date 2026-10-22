import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.model_interface import move_to_action
from go_rl.static_policy import fit_response_frequencies, frequency_move


RULES = "koPOSITIONALscoreAREAtaxNONEsui0"


def training_row(move, board=None):
    return {**state_record_from_board(board or BoardState.empty(3)), "split": "train",
            "rules": RULES, "response": move, "policy_target": [0] * 9 + [1]}


def test_frequency_uses_explicit_responses_and_masks_illegal_most_common_move():
    counts, rows = fit_response_frequencies(
        [training_row("A3"), training_row("A3"), training_row("B2")], board_size=3)
    assert rows == 3 and counts[0] == 2 and counts[4] == 1 and counts[-1] == 0
    board = BoardState.empty(3)
    assert frequency_move(counts, board, rules=RULES) == "A3"
    board.apply_move("B", "A3")
    assert frequency_move(counts, board, rules=RULES) == "B2"


def test_frequency_d4_orbits_and_deterministic_ties():
    counts, _ = fit_response_frequencies([training_row("A3"), training_row("PASS")],
                                         board_size=3, symmetries=8)
    assert counts == [2, 0, 2, 0, 0, 0, 2, 0, 2, 8]
    counts[-1] = 2
    assert frequency_move(counts, BoardState.empty(3), rules=RULES) == "A3"
    assert counts[move_to_action("C1", 3)] == 2


@pytest.mark.parametrize("change,match", [
    ({"split": "validation"}, "training"), ({"response": None}, "explicit"),
    ({"response": "Z99"}, "illegal"),
])
def test_frequency_rejects_nontraining_and_invalid_labels(change, match):
    with pytest.raises(ValueError, match=match):
        fit_response_frequencies([{**training_row("A3"), **change}], board_size=3)


def test_frequency_rejects_occupied_response_and_empty_data():
    board = BoardState.empty(3)
    board.apply_move("B", "A3")
    with pytest.raises(ValueError, match="illegal"):
        fit_response_frequencies([training_row("A3", board)], board_size=3)
    with pytest.raises(ValueError, match="empty"):
        fit_response_frequencies([], board_size=3)
