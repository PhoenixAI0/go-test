import copy
import importlib.util
from pathlib import Path

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.experiment_utils import position_key
from go_rl.rules import normalize_rules
from go_rl.teacher_data import TARGET_RULES

spec = importlib.util.spec_from_file_location(
    "prepare_refresh_ablation", Path(__file__).resolve().parents[1] / "scripts" / "prepare_refresh_ablation.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def teacher_row(board, game, *, learner=False, split="train"):
    return {**state_record_from_board(board), "game_id": game, "split": split,
            "rules": TARGET_RULES, "komi": 7.5, "response": "PASS", "teacher_labeled": True,
            "source_type": "learner_gtp_match" if learner else "archive",
            "teacher": {"max_visits": 64, "engine_model_sha256": "same-network",
                        "engine_config_sha256": "same-config", "normalized_rules": normalize_rules(TARGET_RULES),
                        "target_type": "mixed", "prior_mix": .5}}


def sources():
    expert, learner = [], []
    for game in range(3):
        board = BoardState.empty(19)
        board.apply_move("B", f"A{game+1}", strict=True)
        expert.append(teacher_row(board, f"expert-{game}"))
        board.apply_move("W", "T19", strict=True)
        expert.append(teacher_row(board, f"expert-{game}"))
    for game in range(2):
        board = BoardState.empty(19)
        board.apply_move("B", f"J{game+8}", strict=True)
        learner.append(teacher_row(board, f"learner-{game}", learner=True))
        board.apply_move("W", "Q16", strict=True)
        learner.append(teacher_row(board, f"learner-{game}", learner=True))
    board = BoardState.empty(19)
    board.apply_move("B", "D12", strict=True)
    heldout = [teacher_row(board, "heldout", split="validation")]
    return expert, learner, [heldout]


def test_matched_budget_shared_expert_prefix_and_balanced_learner_games():
    expert, learner, heldouts = sources()
    original = copy.deepcopy((expert, learner, heldouts))
    arms, summary = preparation.prepare_arms(expert, learner, heldouts, positions=6, learner_positions=2)
    assert len(arms["expert_control"]) == len(arms["learner_mix"]) == 6
    assert arms["learner_mix"][:4] == arms["expert_control"][:4]
    assert summary["learner_selected"] == 2 and sorted(summary["learner_game_counts"].values()) == [1, 1]
    assert summary["arms"]["expert_control"]["effective_games_by_position_weight"] == 3
    assert (expert, learner, heldouts) == original
    changed = [{**row, "response": "Q10"} for row in learner]
    _, other = preparation.prepare_arms(expert, changed, heldouts, positions=6, learner_positions=2)
    assert summary["arms"]["learner_mix"]["selected_positions"] == other["arms"]["learner_mix"]["selected_positions"]


def test_small_learner_source_retained_in_full_without_changing_total_budget():
    expert, learner, heldouts = sources()
    arms, summary = preparation.prepare_arms(expert, learner[:1], heldouts, positions=6, learner_positions=2)
    assert summary["learner_selected"] == 1 and summary["shared_expert_positions"] == 5
    assert arms["learner_mix"][-1] == learner[0] and len(arms["learner_mix"]) == 6


def test_mismatched_teacher_budget_or_network_and_unlabeled_sources_fail():
    expert, learner, heldouts = sources()
    for field, value, message in (("max_visits", 8, "64-visit"), ("engine_model_sha256", "other", "same teacher")):
        changed = copy.deepcopy(learner)
        changed[0]["teacher"][field] = value
        with pytest.raises(ValueError, match=message):
            preparation.prepare_arms(expert, changed, heldouts, positions=6, learner_positions=2)
    learner[0]["teacher_labeled"] = False
    with pytest.raises(ValueError, match="teacher-labeled"):
        preparation.prepare_arms(expert, learner, heldouts, positions=6, learner_positions=2)


def test_heldout_state_and_game_leakage_fail_without_resampling():
    expert, learner, heldouts = sources()
    heldouts[0].append({**expert[0], "split": "validation", "game_id": "different-game"})
    with pytest.raises(ValueError, match="state leakage"):
        preparation.prepare_arms(expert, learner, heldouts, positions=6, learner_positions=2)
    heldouts[0][-1]["game_id"] = expert[0]["game_id"]
    with pytest.raises(ValueError, match="game leakage"):
        preparation.prepare_arms(expert, learner, heldouts, positions=6, learner_positions=2)


def test_phase_matched_replacements_are_exact_distinct_and_reproducible():
    def row(game, number):
        return {"game_id": game, "move_number": number, "prompt": f"{game}-{number}", "rules": TARGET_RULES, "komi": 7.5}
    expert = [row(f"expert-{game}", number) for game in range(3) for number in (0, 10, 30, 90, 200, 300)]
    shared = expert[:6]
    learner = [row("learner", number) for number in (0, 10, 30, 90, 200, 300)]
    chosen, audit = preparation.phase_matched_replacements(expert + [expert[-1]], shared, learner, seed=3407)
    assert len(chosen) == len({position_key(row) for row in chosen}) == 6
    assert not {position_key(row) for row in chosen} & {position_key(row) for row in shared}
    assert audit["exact_phase_match"] and not audit["shortages_before_fallback"] and not audit["fallback_positions"]
    assert audit["requested_bins"] == audit["selected_bins"]
    again, _ = preparation.phase_matched_replacements(list(reversed(expert)), shared, learner, seed=3407)
    assert chosen == again


def test_phase_shortage_is_explicit_and_fallback_does_not_repeat_rows():
    expert, learner, _ = sources()
    learner = [{**learner[0], "move_number": 0}]
    chosen, audit = preparation.phase_matched_replacements(expert, expert[:4], learner, seed=3407)
    assert len(chosen) == 1 and chosen[0] not in expert[:4]
    assert audit["shortages_before_fallback"] == {"0": 1}
    assert audit["fallback_positions"] == 1 and audit["repeated_exact_rows"] == 0
    assert not audit["exact_phase_match"]


def test_final_test_cannot_be_used_for_checkpoint_selection_or_overlap_validation():
    _, _, heldouts = sources()
    validation = heldouts[0]
    board = BoardState.empty(19)
    board.apply_move("B", "J10", strict=True)
    test = [teacher_row(board, "test", split="test")]
    preparation.validate_holdout_roles(validation, test)
    with pytest.raises(ValueError, match="pinned to validation"):
        preparation.validate_holdout_roles(test, validation)
    with pytest.raises(ValueError, match="game leakage"):
        preparation.validate_holdout_roles(validation, [{**test[0], "game_id": validation[0]["game_id"]}])
    with pytest.raises(ValueError, match="state leakage"):
        preparation.validate_holdout_roles(validation, [{**validation[0], "game_id": "other", "split": "test"}])
