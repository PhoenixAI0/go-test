import importlib.util
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "audit_terminal_scores", Path(__file__).resolve().parents[1] / "scripts/audit_terminal_scores.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _sgf(body, rules="koPOSITIONALscoreAREAtaxNONEsui0", komi=7.5):
    return f"(;GM[1]FF[4]SZ[3]KM[{komi}]RU[{rules}]{body})".encode()


def test_finished_result_is_preserved_when_raw_area_winner_differs():
    row = runner.audit_sgf(_sgf("RE[W+7.5];B[aa];W[];B[]"), source="finished.sgf")
    assert row["status"] == "compared"
    assert row["original_result"] == row["adjudicated_result"] == "W+7.5"
    assert row["adjudicated_black_outcome"] == -1
    assert row["raw_board_result"] == "B+1.5"
    assert row["adjudicated_minus_raw_black_margin"] == -9
    assert not row["winner_matches"] and not row["margin_matches"]
    assert row["raw_board_area"]["black_stones"] == 1
    assert row["raw_board_area"]["black_territory"] == 8


def test_setup_and_passes_support_draw_and_coordinate_orientation():
    row = runner.audit_sgf(_sgf("AB[aa]PL[W]RE[0];W[];B[]", komi=9), source="draw.sgf")
    assert row["raw_board_result"] == "0" and row["margin_matches"]
    assert row["winner_matches"] and row["plies"] == 2


@pytest.mark.parametrize("body,status", [
    ("RE[Void]C[termination=move_cap];B[aa]", "move_cap"),
    ("RE[Void];B[];W[]", "unscored"),
    ("RE[B+R];B[aa]", "nonnumeric_adjudication"),
    ("RE[B+5];B[aa]", "not_two_pass_completion"),
])
def test_caps_and_incomplete_games_never_get_raw_results(body, status):
    row = runner.audit_sgf(_sgf(body), source="incomplete.sgf")
    assert row["status"] == status and row["raw_board_result"] is None
    if status in {"move_cap", "unscored"}:
        assert row["adjudicated_black_outcome"] is None


def test_named_chinese_rules_are_not_silently_changed_to_raw_area_objective():
    row = runner.audit_sgf(_sgf("RE[W+7.5];B[];W[]", rules="Chinese"), source="chinese.sgf")
    assert row["status"] == "unsupported_raw_area_rules"
    assert row["original_result"] == "W+7.5" and row["raw_board_result"] is None


@pytest.mark.parametrize("body,message", [
    ("RE[B+5]C[termination=move_cap];B[aa]", "move-capped"),
    ("RE[Void];B[aa];W[aa]", "Illegal"),
    ("RE[Void];B[aa];B[bb]", "alternate"),
    ("RE[Void];B[];W[];B[aa]", "two-pass"),
    ("RE[Void];B[aa];AB[bb]", "Mid-game"),
    ("RE[Void];B[aa];PL[B]", "Mid-game"),
    ("RE[Void];B[aa](;W[bb])(;W[cc])", "variations"),
    ("RE[Void];B[aa]W[bb]", "two moves"),
])
def test_invalid_match_history_fails_instead_of_producing_diagnostics(body, message):
    with pytest.raises(ValueError, match=message):
        runner.audit_sgf(_sgf(body), source="invalid.sgf")


def test_cli_report_preserves_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "completed.sgf"
    data = _sgf("RE[W+7.5];B[aa];W[];B[]")
    source.write_bytes(data)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--sgf", str(source), "--output", str(output)])
    runner.main()
    report = json.loads(output.read_text())
    assert report["engine_or_model_queries"] == 0
    assert report["counts"] == {"compared": 1} and report["different_winners"] == 1
    assert source.read_bytes() == data
    with pytest.raises(SystemExit):
        runner.main()
