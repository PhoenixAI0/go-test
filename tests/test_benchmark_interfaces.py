import pytest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark_interfaces_test_module", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_interfaces.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_game_balanced = module.select_game_balanced
teacher_metrics = module.teacher_metrics
summarize = module.summarize


def _row(game, move):
    return {"game_id": game, "move_number": move, "prompt": f"{game}:{move}",
            "rules": "chinese", "komi": 7.5}


def test_sampling_balances_games_and_is_stable_after_reordering_or_duplication():
    rows = [_row(game, move) for game, length in [("a", 50), ("b", 3), ("c", 8)] for move in range(length)]
    sample = select_game_balanced(rows, 9, seed=19)
    assert {game: sum(row["game_id"] == game for row in sample) for game in "abc"} == {"a": 3, "b": 3, "c": 3}
    assert select_game_balanced(list(reversed(rows)) + rows[:4], 9, seed=19) == sample
    assert len(select_game_balanced(rows, 1000)) == len(rows)


def test_teacher_regret_has_correct_perspective_and_unknown_move_coverage():
    row = {"teacher": {"perspective": "black"}, "teacher_best_move": "A1",
           "teacher_move_infos": [{"move": "A1", "winrate": 0.8, "scoreLead": 5},
                                  {"move": "B2", "winrate": 0.3, "scoreLead": -2}]}
    black = teacher_metrics(row, "B2", "B")
    assert black["teacher_winrate_regret_points"] == pytest.approx(50)
    assert black["teacher_score_loss_points"] == pytest.approx(7)
    white = teacher_metrics(row, "A1", "W")
    assert white["teacher_winrate_regret_points"] == pytest.approx(50)
    assert white["teacher_score_loss_points"] == pytest.approx(7)
    unknown = teacher_metrics(row, "C3", "B")
    assert unknown["teacher_chosen_move_evaluated"] is False
    assert "teacher_winrate_regret_points" not in unknown
    row["teacher"]["perspective"] = "unspecified"
    with pytest.raises(ValueError, match="perspective"):
        teacher_metrics(row, "A1", "B")


def test_summary_counts_failures_and_exposes_regret_coverage():
    rows = [
        {"encoding": "grid", "decoding": "constrained", "game_id": "a", "legal": True,
         "seconds": 1.5, "teacher_chosen_move_evaluated": True, "teacher_winrate_regret_points": 4.0},
        {"encoding": "grid", "decoding": "constrained", "game_id": "b", "error": "context too long"},
    ]
    result = summarize(rows)["grid/constrained"]
    assert result["positions"] == 2 and result["errors"] == 1
    assert result["legal_rate"] == 0.5
    assert result["teacher_regret_coverage"] == 0.5
    assert result["mean_teacher_winrate_regret_points"] == 4
    assert result["n_teacher_winrate_regret_points"] == 1
