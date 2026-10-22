import copy

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import board_from_example, state_record_from_board, transform_teacher_example
from go_rl.model_interface import action_to_move, legal_action_mask, move_to_action


def _teacher_record():
    board = BoardState.empty(19)
    for player, move in [("B", "A1"), ("W", "J9"), ("B", "T18")]:
        board.apply_move(player, move, strict=True)
    legal = legal_action_mask(board)
    weights = [float(index + 1) if valid else 0. for index, valid in enumerate(legal)]
    total = sum(weights)
    return {
        **state_record_from_board(board), "game_id": "same-game", "split": "train", "symmetry": 0,
        "rules": "koPOSITIONALscoreAREAtaxNONEsui0", "komi": 7.5,
        "response": "A19", "teacher_best_move": "A19", "source_response": "B3",
        "source_opponent_response": "PASS", "teacher_value": -.75,
        "learner_move": "B3",
        "policy_target": [value / total for value in weights], "legal_mask": legal,
        "ownership": list(range(361)), "ownership_stdev": [value / 361 for value in range(361)],
        "teacher": {"perspective": "black", "value_perspective": "side_to_move"},
        "teacher_move_infos": [{"move": action_to_move(index), "visits": index,
                                "winrate": .25, "scoreLead": -3.5,
                                "isSymmetryOf": action_to_move(index),
                                "pv": [action_to_move(index), "PASS"]} for index in range(362)],
    }


@pytest.mark.parametrize("symmetry", range(8))
def test_teacher_symmetries_align_every_action_history_mask_and_inverse(symmetry):
    original = _teacher_record()
    snapshot = copy.deepcopy(original)
    transformed = transform_teacher_example(original, symmetry)
    assert original == snapshot
    board = board_from_example(transformed)
    assert transformed["legal_mask"] == legal_action_mask(board)
    assert transformed["game_id"] == original["game_id"]
    assert transformed["split"] == "train"
    assert transformed["teacher_value"] == original["teacher_value"]
    assert transformed["teacher"] == original["teacher"]
    assert transformed["komi"] == original["komi"]
    assert transformed["rules"] == original["rules"]
    assert transformed["policy_target"][-1] == original["policy_target"][-1]
    assert transformed["teacher_move_infos"][-1]["move"] == "PASS"
    assert transformed["source_opponent_response"] == "PASS"
    assert transformed["learner_move"] == transformed["source_response"]
    for action, info in enumerate(transformed["teacher_move_infos"]):
        destination = move_to_action(info["move"])
        assert transformed["policy_target"][destination] == original["policy_target"][action]
        assert transformed["legal_mask"][destination] == original["legal_mask"][action]
        assert info["pv"] == [info["move"], "PASS"]
        assert info["isSymmetryOf"] == info["move"]
        assert info["winrate"] == .25 and info["scoreLead"] == -3.5
        if action < 361:
            assert transformed["ownership"][destination] == original["ownership"][action]
    assert transformed["response"] == transformed["teacher_move_infos"][0]["move"]
    assert transformed["teacher_best_move"] == transformed["response"]
    inverse = (-symmetry) % 4 if symmetry < 4 else symmetry
    assert transform_teacher_example(transformed, inverse) == original
    transformed["teacher"]["perspective"] = "changed"
    assert original["teacher"]["perspective"] == "black"


def test_teacher_symmetry_rejects_spatial_shape_mismatch():
    record = _teacher_record()
    record["policy_target"].pop()
    with pytest.raises(ValueError, match="362 entries"):
        transform_teacher_example(record, 0)
