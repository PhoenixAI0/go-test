import math

import pytest

from go_rl.board import BoardState, coords_to_gtp
from go_rl.data_prep import state_record_from_board
from go_rl.katago import KataGoResult
from go_rl.move_quality import (
    filter_preterminal_records, game_cluster_interval, score_move_predictions, summarize_move_quality,
)


def record(player, game_id):
    board = BoardState.from_matrix(3, [[0] * 3 for _ in range(3)], next_player=player)
    return {**state_record_from_board(board), "game_id": game_id, "split": "validation",
            "rules": "koPOSITIONALscoreAREAtaxNONEsui0", "komi": 7.5}


class FakeAnalysis:
    def __init__(self):
        self.queries = []

    def analyze_positions(self, boards, **options):
        results = []
        for board in boards:
            move = coords_to_gtp(board.moves[-1][1], board.size) if board.moves else None
            self.queries.append((board.next_player(), move, options))
            score, winrate = {None: (10., .8), "A3": (7., .7), "C1": (-2., .2)}.get(move, (0., .5))
            # C1 is intentionally absent from root moveInfos; evaluator must query its child.
            results.append(KataGoResult(winrate, {"rootInfo": {"visits": options["max_visits"]},
                                                   "moveInfos": [{"move": "A3"}]},
                                        score_lead=score, best_move="A3"))
        return results


def test_unsearched_model_moves_get_actual_equal_budget_child_evaluations_and_player_sign():
    engine = FakeAnalysis()
    rows, counts = score_move_predictions(
        [record("B", "black"), record("W", "white")],
        {"model": [{"move": "C1"}, {"move": "C1"}], "same": [{"move": "C1"}, {"move": "C1"}]},
        engine, max_visits=32, batch_size=2,
    )
    assert rows[0]["agents"]["model"]["score_difference_vs_reference"] == -9
    assert rows[1]["agents"]["model"]["score_difference_vs_reference"] == 9
    assert rows[0]["agents"]["model"]["winrate_difference_vs_reference"] == pytest.approx(-.5)
    assert rows[1]["agents"]["model"]["winrate_difference_vs_reference"] == pytest.approx(.5)
    assert all(options["max_visits"] == 32 for _, _, options in engine.queries)
    assert len([move for _, move, _ in engine.queries if move == "C1"]) == 2  # Shared model choices cached.
    assert counts["root_queries"] == 2 and counts["total_queries"] == len(engine.queries)
    assert all(row["agents"]["reference"]["score_difference_vs_reference"] == 0 for row in rows)


def test_invalid_model_move_is_reported_without_invented_child_score():
    engine = FakeAnalysis()
    rows, _ = score_move_predictions([record("B", "g")], {"bad": [{"move": "Z99", "text": "invalid"}]}, engine)
    prediction = rows[0]["agents"]["bad"]
    assert not prediction["legal"] and prediction["child_estimate"] is None
    assert "score_difference_vs_reference" not in prediction
    summary = summarize_move_quality(rows, bootstrap_samples=20)["bad"]
    assert summary["legal"]["mean"] == 0
    assert summary["score_difference_vs_reference"]["mean"] is None
    assert all(move != "Z99" for _, move, _ in engine.queries)


def test_uniform_draw_is_reproducible_per_position_even_when_input_order_changes():
    records = [record("B", "a"), record("W", "b")]
    before, _ = score_move_predictions(records, {"model": [{"move": "C1"}] * 2}, FakeAnalysis(), seed=9)
    after, _ = score_move_predictions(records[::-1], {"model": [{"move": "C1"}] * 2}, FakeAnalysis(), seed=9)
    assert {row["game_id"]: row["agents"]["uniform"]["move"] for row in before} == {
        row["game_id"]: row["agents"]["uniform"]["move"] for row in after}


def test_game_cluster_interval_weights_games_equally_and_requires_multiple_games_for_ci():
    observations = [("a", 0.), ("a", 0.), ("a", 0.), ("b", 1.)]
    result = game_cluster_interval(observations, samples=500, seed=8)
    assert result["mean"] == .5 and result["positions"] == 4 and result["games"] == 2
    assert result["ci95"] == [0., 1.]
    assert result == game_cluster_interval(observations[::-1], samples=500, seed=8)
    assert game_cluster_interval([("a", 1.), ("a", 0.)])["ci95"] is None
    with pytest.raises(ValueError, match="finite"):
        game_cluster_interval([("a", math.nan)])


def test_missing_engine_score_does_not_silently_become_zero():
    class MissingScore(FakeAnalysis):
        def analyze_positions(self, boards, **options):
            results = super().analyze_positions(boards, **options)
            results[0].score_lead = None
            return results

    with pytest.raises(ValueError, match="scoreLead"):
        score_move_predictions([record("B", "a")], {"model": [{"move": "C1"}]}, MissingScore())


def test_terminal_and_past_terminal_histories_excluded_before_queries():
    board = BoardState.empty(3)
    board.apply_move("B", "PASS")
    single_pass = {**record("B", "single"), **state_record_from_board(board)}
    board.apply_move("W", "PASS")
    terminal = {**record("B", "terminal"), **state_record_from_board(board)}
    board.apply_move("B", "A3")
    past_terminal = {**record("B", "past"), **state_record_from_board(board)}
    accepted, rejected = filter_preterminal_records([terminal, single_pass, past_terminal])
    assert accepted == [single_pass]
    assert [row["source_row"] for row in rejected] == [0, 2]
    assert all(row["reason"] == "history_reaches_two_pass_termination" for row in rejected)
    engine = FakeAnalysis()
    with pytest.raises(ValueError, match="precede two-pass"):
        score_move_predictions([past_terminal], {"model": [{"move": "C1"}]}, engine)
    assert engine.queries == []


def test_posthoc_terminal_audit_preserves_source_and_saved_estimates(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "terminal_audit_test", Path(__file__).resolve().parents[1] / "scripts/audit_move_quality_terminal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows, counts = score_move_predictions([record("B", "valid"), record("W", "old_terminal")],
                                          {"model": [{"move": "C1"}] * 2}, FakeAnalysis())
    board = BoardState.empty(3)
    board.apply_move("B", "PASS")
    board.apply_move("W", "PASS")
    rows[1]["position"].update(state_record_from_board(board))
    source = tmp_path / "original.json"
    source.write_text(json.dumps({"positions": 2, "games": 2, "details": rows, "queries": counts,
                                 "seconds": 12.5, "arguments": {"seed": 9, "bootstrap_samples": 20}}))
    original = source.read_bytes()
    result = module.audit_report(source, tmp_path / "audited.json")
    assert source.read_bytes() == original
    assert result["details"] == [rows[0]]
    assert result["queries"] == counts and result["seconds"] == 12.5
    assert result["post_hoc_terminal_audit"]["new_engine_or_model_queries"] == 0
    assert result["positions"] == 1 and result["summary"]["model"]["legal"]["positions"] == 1
