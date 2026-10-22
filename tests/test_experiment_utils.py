import math

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board, transform_example
from go_rl.experiment_utils import (policy_baselines, state_identity, validate_splits,
                                    select_game_balanced, select_game_balanced_indices)


def _record(game_id):
    board = BoardState(5)
    board.apply_move("B", "B2", strict=True)
    return {**state_record_from_board(board), "game_id": game_id,
            "response": "C3", "rules": "Chinese", "komi": 7.5}


def test_isolation_detects_distinct_games_with_rotated_visible_states():
    original = _record("train")
    rotated = transform_example(_record("test"), 5)
    assert state_identity(original) == state_identity(rotated)
    with pytest.raises(ValueError, match="state leakage"):
        validate_splits([original], [rotated])
    with pytest.raises(ValueError, match="game leakage"):
        validate_splits([original], [original])
    rotated["komi"] = 6.5
    validate_splits([original], [rotated])


def test_baselines_fit_only_training_targets_and_mask_frequencies():
    train = [{"legal": [True, True, False], "target": [.2, .8, 0.], "action": 1}]
    evaluation = [{"legal": [True, True, False], "target": [.7, .3, 0.], "action": 0},
                  {"legal": [True, False, False], "target": [1., 0., 0.], "action": 0}]
    result = policy_baselines(evaluation, train)
    assert result["uniform"]["policy_loss"] == pytest.approx(math.log(2) / 2)
    assert result["uniform"]["top1_expected"] == .75
    assert result["training_frequency"]["top1"] == .5
    expected = (-.7 * math.log((.2 + 1e-6) / (1 + 2e-6))
                - .3 * math.log((.8 + 1e-6) / (1 + 2e-6))) / 2
    assert result["training_frequency"]["policy_loss"] == pytest.approx(expected)


def test_baselines_reject_illegal_probability_mass():
    with pytest.raises(ValueError, match="illegal"):
        policy_baselines([{"legal": [True, False], "target": [.5, .5], "action": 0}])


@pytest.mark.parametrize("count", [1, 4, 13, 100])
def test_index_selector_matches_exact_stable_selection_with_duplicates(count):
    rows = [{"game_id": f"game-{game}", "prompt": f"position-{position}", "move_number": position,
             "rules": "Chinese", "komi": 7.5, "response": "A1"}
            for game in range(7) for position in range(game + 1)]
    # Position-key duplicates preserve the last row's ancillary target fields.
    rows += [{**rows[3], "response": "PASS"}, dict(rows[0])]
    for seed in (1, 42, 3407):
        expected = select_game_balanced(rows, count, seed)
        assert [rows[index] for index in select_game_balanced_indices(rows, count, seed)] == expected


@pytest.mark.parametrize("pinned", ["validation", "test", "val", "unknown"])
def test_training_rejects_pinned_heldout_or_unknown_partitions(pinned):
    train, validation = _record("train"), _record("validation")
    train["split"] = pinned
    validation["komi"] = 6.5
    with pytest.raises(ValueError, match="Training.*pinned"):
        validate_splits([train], [validation])


@pytest.mark.parametrize("pinned", ["train", "unknown"])
def test_evaluation_rejects_pinned_training_or_unknown_partitions(pinned):
    train, validation = _record("train"), _record("validation")
    validation.update(komi=6.5, split=pinned)
    with pytest.raises(ValueError, match="Evaluation.*pinned"):
        validate_splits([train], [validation])


@pytest.mark.parametrize("train_split", [None, "", "train"])
@pytest.mark.parametrize("eval_split", [None, "", "validation", "test"])
def test_pinned_partitions_and_unpinned_legacy_records_are_accepted(train_split, eval_split):
    train, validation = _record("train"), _record("validation")
    train["split"] = train_split
    validation.update(komi=6.5, split=eval_split)
    validate_splits([train], [validation])


@pytest.mark.parametrize("eval_split", [None, "", "validation", "test"])
def test_checkpoint_selection_excludes_final_test_but_preserves_unpinned_legacy(eval_split):
    train, validation = _record("train"), _record("validation")
    validation.update(komi=6.5, split=eval_split)
    if eval_split == "test":
        with pytest.raises(ValueError, match="Evaluation.*pinned.*test"):
            validate_splits([train], [validation], allow_test=False)
        validate_splits([train], [validation])  # Standalone final evaluation remains supported.
    else:
        validate_splits([train], [validation], allow_test=False)
