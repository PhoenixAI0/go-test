import importlib.util
import json
from pathlib import Path

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.experiment_utils import file_sha256, read_records
from go_rl.teacher_data import TARGET_RULES

spec = importlib.util.spec_from_file_location(
    "filter_terminal_positions", Path(__file__).resolve().parents[1] / "scripts" / "filter_terminal_positions.py")
filtering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filtering)


def test_subset_retains_existing_labels_and_objective_exactly(tmp_path):
    board = BoardState.empty(3)
    rows = []
    for move in (None, "PASS", "PASS", "A1"):
        if move:
            board.apply_move(board.next_player(), move, strict=True)
        rows.append({**state_record_from_board(board), "game_id": "one", "split": "validation",
                     "rules": TARGET_RULES, "komi": 7.5, "teacher_labeled": True,
                     "policy_target": [0.1] * 10, "teacher_value": .123, "response": "PASS",
                     "source_result": "B+R", "teacher": {"engine_model_sha256": "existing-engine"}})
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = tmp_path / "source.manifest.json"
    manifest_path.write_text(json.dumps({"output_sha256": file_sha256(source)}))
    source_bytes = source.read_bytes()
    target = tmp_path / "target.jsonl"
    manifest = filtering.freeze_preterminal_subset(source, manifest_path, target, split="validation")
    assert read_records(target) == rows[:2]
    assert source.read_bytes() == source_bytes
    assert manifest["rejection_count"] == 2 and manifest["retained_source_rows"] == [0, 1]
    assert manifest["labels_and_objective_unchanged"] is True
    assert manifest["new_engine_or_model_queries"] is False
    with pytest.raises(ValueError, match="fresh path"):
        filtering.freeze_preterminal_subset(source, manifest_path, target, split="validation")
    with pytest.raises(ValueError, match="pinned split"):
        filtering.freeze_preterminal_subset(source, manifest_path, tmp_path / "wrong.jsonl", split="train")
