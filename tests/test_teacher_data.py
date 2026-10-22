import pytest

from go_rl.board import BoardState
from go_rl.katago import KataGoResult
from go_rl.teacher_data import make_teacher_record, select_game_balanced, select_phase_balanced, move_phase


def analysis():
    return KataGoResult(.75, {"policy": [.1] * 10,
                            "moveInfos": [{"move": "A3", "visits": 9}, {"move": "PASS", "visits": 1}]},
                        score_lead=4, best_move="A3")


def test_teacher_relabel_preserves_source_and_flips_value_for_white():
    board = BoardState.from_matrix(3, [[0] * 3 for _ in range(3)], "W")
    source = {"rules": "japanese", "komi": 6.5, "response": "B2", "result": "W+R", "opponent_response": "C3"}
    record = make_teacher_record(source, board, analysis(), prior_mix=.5)
    assert record["source_rules"] == "japanese" and record["source_result"] == "W+R"
    assert "result" not in record and "opponent_response" not in record
    assert record["teacher_value"] == -.5 and record["response"] == "A3"
    assert record["teacher"]["perspective"] == "black"
    assert record["teacher"]["value_perspective"] == "side_to_move"
    assert record["teacher"]["schema_version"] == 2
    assert record["policy_target"][0] == pytest.approx(.5)
    assert sum(record["policy_target"]) == pytest.approx(1)


def test_prior_target_and_searched_best_move_are_distinct():
    result = analysis()
    result.raw["policy"] = [.01] * 10
    result.raw["policy"][4] = .91
    record = make_teacher_record({}, BoardState.empty(3), result, prior_mix=1)
    assert record["response"] == "A3"  # Search winner, not raw-prior argmax B2.
    assert max(range(10), key=record["policy_target"].__getitem__) == 4
    assert record["teacher"]["target_type"] == "prior"


def test_teacher_rejects_legality_disagreement():
    board = BoardState.empty(3)
    result = analysis()
    result.raw["policy"][0] = -1
    with pytest.raises(ValueError, match="masks disagree"):
        make_teacher_record({}, board, result)


def test_game_sampling_is_bounded_balanced_and_reproducible():
    records = [{"game_id": str(game), "move_number": turn} for game in range(6) for turn in range(10)]
    selected = select_game_balanced(records, 8, 42, per_game=2)
    assert selected == select_game_balanced(records, 8, 42, per_game=2)
    assert len(selected) == 8
    assert len({record["game_id"] for record in selected[:6]}) == 6


def test_phase_sampling_keeps_ends_balances_games_and_ignores_input_order():
    records = [{"game_id": str(game), "move_number": turn}
               for game in range(6) for turn in (10, 20, 80, 120, 200, 250, 300, 350)]
    selected = select_phase_balanced(records, 24, 42, per_game=4)
    assert selected == select_phase_balanced(list(reversed(records)) + records, 24, 42, per_game=4)
    assert len(selected) == 24
    assert len({row["game_id"] for row in selected[:6]}) == 6
    for game in range(6):
        assert {move_phase(row["move_number"]) for row in selected if row["game_id"] == str(game)} == {
            "0-60", "61-180", "181-280", "281+"}


def test_phase_sampling_handles_missing_and_exhausted_phases_without_using_labels():
    records = [{"game_id": "only", "move_number": turn, "response": "PASS"}
               for turn in (10, 20, 30, 310, 320)]
    selected = select_phase_balanced(records, 20, 17, per_game=8)
    assert len(selected) == 5
    changed = [{**row, "response": "D4"} for row in records]
    assert [row["move_number"] for row in selected] == [
        row["move_number"] for row in select_phase_balanced(changed, 20, 17, per_game=8)]
    with pytest.raises(ValueError, match="integer"):
        select_phase_balanced([{"game_id": "x"}], 1, 42)
