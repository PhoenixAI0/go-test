import copy
import importlib.util
import json
from pathlib import Path

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.experiment_utils import file_sha256
from go_rl.teacher_data import TARGET_RULES

spec = importlib.util.spec_from_file_location(
    "prepare_eval_objective", Path(__file__).resolve().parents[1] / "scripts" / "prepare_eval_objective.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def record(board=None):
    return {**state_record_from_board(board or BoardState.empty(3)), "split": "test", "game_id": "game-a",
            "rules": "japanese", "komi": 6.5, "response": "B2", "opponent_response": "A1",
            "result": "W+R", "source_file": "archive#original.sgf"}


def prepare(rows):
    return preparation.target_test_records(rows, rules=TARGET_RULES, komi=7.5, board_size=3,
                                           source_sha256="frozen-source-hash")


def test_target_preserves_source_and_never_selects_on_move_or_outcome_labels():
    source = record()
    before = copy.deepcopy(source)
    changed = {**source, "response": "not a move", "result": "not a result", "opponent_response": "PASS"}
    rows, rejected = prepare([source, changed])
    assert not rejected and len(rows) == 2
    assert source == before
    for index, row in enumerate(rows):
        assert row["rules"] == TARGET_RULES and row["komi"] == 7.5
        assert row["source_rules"] == "japanese" and row["source_komi"] == 6.5
        assert row["source_file"] == source["source_file"]
        assert row["split"] == "test" and row["teacher_labeled"] is False
        assert not {"response", "result", "opponent_response", "policy_target", "teacher_value"} & row.keys()
        assert row["evaluation_target"]["source_row"] == index
        assert row["evaluation_target"]["engine_or_model_evaluated"] is False
    assert rows[0]["source_response"] == "B2" and rows[1]["source_response"] == "not a move"


def test_terminal_and_corrupt_histories_are_rejected_without_resampling():
    board = BoardState.empty(3)
    board.apply_move("B", "PASS", strict=True)
    one_pass = record(board)
    board.apply_move("W", "PASS", strict=True)
    terminal = record(board)
    board.apply_move("B", "A1", strict=True)
    continued = record(board)
    bad_count = {**one_pass, "move_number": 0}
    bad_state = {**one_pass, "prompt": record()["prompt"]}
    accepted, rejected = prepare([one_pass, terminal, continued, bad_count, bad_state])
    assert len(accepted) == 1 and accepted[0]["moves"] == [["B", "PASS"]]
    assert [row["source_row"] for row in rejected] == [1, 2, 3, 4]
    assert [row["category"] for row in rejected] == ["terminal_history"] * 2 + ["invalid_state_history"] * 2


def test_target_rule_incompatibility_is_recorded(monkeypatch):
    def incompatible(board, rules):
        raise ValueError("Source history violates target positional superko")
    monkeypatch.setattr(preparation, "validate_history_for_rules", incompatible)
    accepted, rejected = prepare([record()])
    assert not accepted and rejected[0]["category"] == "incompatible_target_rules"


@pytest.mark.parametrize("changes", [{"split": "train"}, {"teacher_labeled": True}, {"policy_target": [1]}])
def test_training_or_already_labeled_sources_fail(changes):
    with pytest.raises(ValueError):
        prepare([{**record(), **changes}])


def test_freeze_checks_manifest_and_preserves_source_bytes(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record()) + "\n")
    manifest_path = tmp_path / "source.manifest.json"
    manifest = {"split": "test", "board_size": 3, "positions": 1, "games": 1, "game_ids": ["game-a"],
                "dataset_sha256": file_sha256(source), "engine_or_model_evaluated": False}
    manifest_path.write_text(json.dumps(manifest))
    before = source.read_bytes(), manifest_path.read_bytes()
    output = tmp_path / "target.jsonl"
    frozen = preparation.freeze_target(source, manifest_path, output)
    assert frozen["dataset_sha256"] == file_sha256(output)
    assert frozen["source_dataset_sha256"] == manifest["dataset_sha256"]
    assert frozen["source_manifest"] == manifest
    assert frozen["retained_source_rows"] == [0]
    assert frozen["source_outcomes_apply_to_target"] is False
    assert before == (source.read_bytes(), manifest_path.read_bytes())
    with pytest.raises(ValueError, match="fresh path"):
        preparation.freeze_target(source, manifest_path, output)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="hash differs"):
        preparation.freeze_target(source, manifest_path, tmp_path / "new.jsonl")
