import importlib.util
from pathlib import Path
import weakref

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.teacher_data import TARGET_RULES

spec = importlib.util.spec_from_file_location(
    "build_teacher_dataset", Path(__file__).resolve().parents[1] / "scripts" / "build_teacher_dataset.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_teacher_batches_do_not_keep_all_replayed_histories(monkeypatch):
    live = weakref.WeakSet()
    peak = 0

    class CachedBoard:
        size = 19
        moves = []

        def __init__(self):
            nonlocal peak
            live.add(self)
            peak = max(peak, len(live))

    monkeypatch.setattr(builder, "board_from_example", lambda row: CachedBoard())
    monkeypatch.setattr(builder, "validate_history_for_rules", lambda board, rules: None)
    records = [{"game_id": str(i), "moves": [], "initial_stones": [], "move_number": 0}
               for i in range(1000)]
    rejected = []
    count = 0
    for chunk in builder.teacher_batches(records, 8, TARGET_RULES, rejected):
        count += len(chunk)
        chunk.clear()
    assert count == 1000 and not rejected
    assert peak <= 9  # Batch plus the previous generator-local board.


def test_teacher_batches_reject_terminal_and_incomplete_histories():
    board = BoardState.empty(3)
    initial = {**state_record_from_board(board), "game_id": "valid"}
    board.apply_move("B", "PASS", strict=True)
    one_pass = {**state_record_from_board(board), "game_id": "one_pass"}
    board.apply_move("W", "PASS", strict=True)
    terminal = {**state_record_from_board(board), "game_id": "terminal"}
    board.apply_move("B", "A1", strict=True)
    continued = {**state_record_from_board(board), "game_id": "continued"}
    missing = dict(initial)
    del missing["moves"]
    mismatched = {**initial, "move_number": 17}
    rejected = []
    batches = list(builder.teacher_batches(
        [initial, one_pass, terminal, continued, missing, mismatched], 2, TARGET_RULES, rejected))
    assert [row["game_id"] for chunk in batches for row, _ in chunk] == ["valid", "one_pass"]
    assert len(rejected) == 4
    assert sum("two-pass" in row["reason"] for row in rejected) == 2


def test_teacher_batches_reject_mixed_board_sizes():
    records = [state_record_from_board(BoardState.empty(size)) for size in (3, 5)]
    with pytest.raises(ValueError, match="single board size"):
        list(builder.teacher_batches(records, 8, TARGET_RULES, []))
