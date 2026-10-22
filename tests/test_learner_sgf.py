import importlib.util
from collections import Counter
import json
from pathlib import Path
import sys

import pytest

from go_rl.board import BoardState, gtp_to_coords
from go_rl.data_prep import board_from_example
from go_rl.gtp_match import GameRecord, MatchConfig, PlayerConfig, game_to_sgf
from go_rl.learner_sgf import sgf_training_records


LEARNER = "qwen ] text"
MOVES = [("B", "B2"), ("W", "A2"), ("B", "A3"), ("W", "PASS"),
         ("B", "A1"), ("W", "T19"), ("B", "PASS"), ("W", "PASS")]


def _match_sgf(learner_color="B", *, capped=False):
    config = MatchConfig(19, 1, PlayerConfig(LEARNER, "unused"), PlayerConfig("KataGo", "unused"),
                         rules="koPOSITIONALscoreAREAtaxNONEsui0")
    black, white = (LEARNER, "KataGo") if learner_color == "B" else ("KataGo", LEARNER)
    record = GameRecord(black, white, MOVES[:5] if capped else MOVES,
                        "Void" if capped else "W+7.5", "move_cap" if capped else "scored")
    return game_to_sgf(record, config).encode()


@pytest.mark.parametrize("learner_color", ["B", "W"])
def test_real_match_sgf_export_selects_learner_turns_and_preserves_opponent_history(learner_color):
    rows = sgf_training_records(_match_sgf(learner_color), learner_name=LEARNER,
                               source_file="match.sgf", checkpoint={"sha256": "snapshot"})
    assert len(rows) == 4
    expected_board = BoardState.empty(19)
    by_ply = {row["move_number"]: row for row in rows}
    for ply, (player, move) in enumerate(MOVES):
        if player == learner_color:
            row = by_ply[ply]
            replayed = board_from_example(row)
            assert replayed.current_board() == expected_board.current_board()
            assert replayed.moves == expected_board.moves
            assert row["moves"] == [list(item) for item in MOVES[:ply]]
            assert row["to_play"] == learner_color and row["learner_move"] == move
            assert row["learner"]["player_color"] == learner_color
            assert row["learner"]["checkpoint"] == {"sha256": "snapshot"}
            assert {row["learner"]["black_name"], row["learner"]["white_name"]} == {LEARNER, "KataGo"}
            assert row["source_result"] == "W+7.5"
            assert row["learner"]["adjudicated_black_outcome"] == -1.
            assert row["teacher_labeled"] is False and row["split"] == "train"
            assert "response" not in row and "result" not in row
        expected_board.apply_move(player, move, strict=True)
    # This SGF exercises a capture: A2 must be absent in subsequent histories.
    row, col = gtp_to_coords("A2", 19)
    assert board_from_example(rows[-1]).current_board()[row][col] == 0
    assert len({row["game_id"] for row in rows}) == 1


def test_capped_match_keeps_void_provenance_without_a_win_or_expert_label():
    rows = sgf_training_records(_match_sgf(capped=True), learner_name=LEARNER, source_file="cap.sgf")
    assert len(rows) == 3
    for row in rows:
        assert "result" not in row and "source_result" not in row and "teacher_value" not in row
        assert row["learner"]["sgf_result"] == "Void"
        assert row["learner"]["adjudicated_black_outcome"] is None
        assert row["learner"]["match_termination"] == "move_cap"


def test_setup_and_initial_player_are_reconstructed_in_sgf_coordinate_orientation():
    data = b"(;GM[1]FF[4]SZ[3]KM[7.5]RU[Chinese]PB[other]PW[learner]AB[aa]PL[W]RE[Void];W[bb];B[cc])"
    rows = sgf_training_records(data, learner_name="learner", source_file="setup.sgf")
    assert len(rows) == 1
    assert rows[0]["initial_stones"] == [["B", "A3"]]
    assert rows[0]["initial_player"] == "W" and rows[0]["learner_move"] == "B2"
    board_from_example(rows[0])


@pytest.mark.parametrize("body,message", [
    ("RE[B+nan];B[aa]", "winning margin"),
    ("RE[B+1.5]C[termination=move_cap];B[aa]", "move-capped"),
    ("RE[Void];B[aa];W[aa]", "Illegal"),
    ("RE[Void];B[aa];B[bb]", "alternate"),
    ("RE[Void];B[aa];AB[bb]", "Mid-game"),
    ("RE[Void];B[];W[];B[aa]", "two-pass"),
])
def test_invalid_match_provenance_or_history_is_rejected(body, message):
    data = f"(;GM[1]FF[4]SZ[3]KM[7.5]RU[Chinese]PB[learner]PW[other]{body})".encode()
    with pytest.raises(ValueError, match=message):
        sgf_training_records(data, learner_name="learner", source_file="bad.sgf")


def test_exact_player_selection_rejects_accidental_opponent_or_missing_names():
    with pytest.raises(ValueError, match="neither PB nor PW"):
        sgf_training_records(_match_sgf(), learner_name="qwen", source_file="match.sgf")


def test_sgf_cli_writes_replayable_sources_and_refuses_to_overwrite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "learner_sgf_cli_test", Path(__file__).resolve().parents[1] / "scripts/export_learner_sgf.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    directory = tmp_path / "sgfs"
    directory.mkdir()
    (directory / "match.sgf").write_bytes(_match_sgf())
    output = tmp_path / "learner.jsonl"
    monkeypatch.setattr(sys, "argv", ["export", "--sgf-dir", str(directory),
                                     "--learner-name", LEARNER, "--output", str(output)])
    runner.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["input_positions"] == 4 and manifest["exported_positions"] == len(rows)
    assert manifest["sources"][0]["black_name"] == LEARNER
    assert all(row["split"] == "train" and row["to_play"] == "B" for row in rows)
    with pytest.raises(SystemExit):
        runner.main()


def test_sgf_cli_samples_games_only_after_deduplication_and_exclusion(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "learner_sgf_sampling_test", Path(__file__).resolve().parents[1] / "scripts/export_learner_sgf.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    directory = tmp_path / "sgfs"
    directory.mkdir()
    first = _match_sgf()
    for name in ("a.sgf", "a_duplicate.sgf"):
        (directory / name).write_bytes(first)
    config = MatchConfig(19, 1, PlayerConfig(LEARNER, "unused"), PlayerConfig("KataGo", "unused"),
                         rules="koPOSITIONALscoreAREAtaxNONEsui0")
    other = GameRecord(LEARNER, "KataGo", [("B", "C2"), *MOVES[1:]], "W+7.5", "scored")
    (directory / "b.sgf").write_text(game_to_sgf(other, config))
    forbidden = sgf_training_records(first, learner_name=LEARNER, source_file="original")[1]
    exclusion = tmp_path / "heldout.jsonl"
    exclusion.write_text(json.dumps({**forbidden, "split": "validation", "response": "T19"}) + "\n")
    outputs = []
    for repeat in range(2):
        output = tmp_path / f"sample{repeat}.jsonl"
        monkeypatch.setattr(sys, "argv", ["export", "--sgf-dir", str(directory), "--learner-name", LEARNER,
                                         "--output", str(output), "--exclude-dataset", str(exclusion),
                                         "--max-positions", "4", "--per-game", "2", "--seed", "17"])
        runner.main()
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        outputs.append(rows)
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        assert manifest["input_positions"] == 12
        assert manifest["excluded_states"] == 2 and manifest["duplicate_states"] == 4
        assert manifest["eligible_positions_after_dedup"] == 6
        assert manifest["eligible_games_after_dedup"] == 2
        assert manifest["selected_positions"] == manifest["exported_positions"] == len(rows) == 4
        assert manifest["budget_excluded_positions"] == 2
        assert sorted(Counter(row["game_id"] for row in rows).values()) == [2, 2]
        assert manifest["sampling"]["applied_after"] == "deduplication_and_heldout_state_exclusion"
        assert manifest["sampling"]["uses_move_quality_or_outcomes"] is False
        assert all(row["split"] == "train" and row["teacher_labeled"] is False and "response" not in row for row in rows)
        assert not any(row["prompt"] == forbidden["prompt"] for row in rows)
        for row in rows:
            board_from_example(row)
    assert outputs[0] == outputs[1]
