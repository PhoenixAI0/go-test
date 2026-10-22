import random

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.seed_positions import prepare_seed_records, replay_seed_board, sample_training_seed


RULES = "koPOSITIONALscoreAREAtaxNONEsui0"


def record(moves=("A3", "B2", "C1"), game_id="seed"):
    board = BoardState.empty(3)
    for move in moves:
        board.apply_move(board.next_player(), move, strict=True)
    return {**state_record_from_board(board), "game_id": game_id, "split": "train",
            "rules": RULES, "komi": 7.5}


def prepare(rows, **kwargs):
    return prepare_seed_records(rows, board_size=3, rules=RULES, komi=7.5, min_move=2, **kwargs)


def test_seed_order_and_duplicates_do_not_change_candidates_and_labels_are_unused():
    first, second = record(), record(("A2", "B3", "C2"), "second")
    expected, rejected = prepare([first, second])
    actual, _ = prepare([second, first, first])
    assert expected == actual and all(value == 0 for value in rejected.values())
    changed, _ = prepare([{**first, "response": "WRONG", "teacher_value": float("nan")}, second])
    assert [row["game_id"] for row in changed] == [row["game_id"] for row in expected]


@pytest.mark.parametrize("updates,match", [
    ({"split": "test"}, "training"), ({"split": None}, "training"),
    ({"komi": 6.5}, "objective"), ({"rules": "Japanese"}, "objective"),
    ({"move_number": 99}, "move_number"), ({"game_id": ""}, "game_id"),
])
def test_seed_candidates_fail_on_leakage_objective_or_provenance_corruption(updates, match):
    with pytest.raises(ValueError, match=match):
        prepare([{**record(), **updates}])


def test_unsupported_setup_and_terminal_seeds_are_excluded_with_counts():
    setup = BoardState.from_matrix(3, [[1, 0, 0], [0, 0, 0], [0, 0, 0]], "W")
    setup.apply_move("W", "B2", strict=True)
    setup.apply_move("B", "C1", strict=True)
    setup_record = {**state_record_from_board(setup), "game_id": "setup", "split": "train",
                    "rules": RULES, "komi": 7.5}
    terminal = record(("PASS", "PASS"), "terminal")
    before = record(("A3",), "early")
    selected, rejected = prepare([record(), setup_record, terminal, before])
    assert len(selected) == 1
    assert rejected == {"before_minimum_move": 1, "initial_setup": 1, "terminal_history": 1}
    with pytest.raises(ValueError, match="No eligible"):
        prepare([setup_record, terminal])


def test_seed_replay_rejects_size_white_empty_start_and_already_finished_board():
    with pytest.raises(ValueError, match="size"):
        replay_seed_board(BoardState.empty(5), board_size=3, rules=RULES)
    white = BoardState.from_matrix(3, [[0] * 3 for _ in range(3)], "W")
    with pytest.raises(ValueError, match="reconstruct"):
        replay_seed_board(white, board_size=3, rules=RULES)
    terminal = BoardState.empty(3)
    terminal.apply_move("B", "PASS")
    terminal.apply_move("W", "PASS")
    with pytest.raises(ValueError, match="already terminal"):
        replay_seed_board(terminal, board_size=3, rules=RULES)


def test_seed_sampling_weights_games_equally_despite_unequal_position_counts():
    records = [{"game_id": "small", "prompt": "one", "move_number": 1},
               *[{"game_id": "large", "prompt": str(i), "move_number": i} for i in range(20)]]
    rng = random.Random(3407)
    selected = [sample_training_seed(records, rng)["game_id"] for _ in range(1000)]
    assert 430 < selected.count("small") < 570
    other = random.Random(3407)
    assert selected == [sample_training_seed(list(reversed(records)), other)["game_id"] for _ in range(1000)]
