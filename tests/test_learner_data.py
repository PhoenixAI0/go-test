import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import board_from_example, transform_example
from go_rl.learner_data import deduplicate_training_records, episode_training_records
from go_rl.model_interface import action_to_move


RUN = {"board_size": 3, "rules": "koPOSITIONALscoreAREAtaxNONEsui0", "komi": 7.5,
       "seed": 42, "model": "immutable/start/checkpoint", "temperature": .8}


def _episode(moves=None, *, completed=False):
    moves = moves or [["B", "A3"], ["W", "B2"], ["B", "C1"]]
    return {"moves": moves, "plies": len(moves), "update": 2, "episode": 0,
            "terminated": completed, "truncated": not completed,
            "black_outcome": -1. if completed else None,
            "final_score": "W+7.5" if completed else None,
            "outcome_source": "katago_gtp_final_score" if completed else "value_bootstrap_at_cap",
            "behavior_temperature": .8}


def _records(episode=None):
    return episode_training_records(episode or _episode(), RUN, source_run_id="run-1", episode_row=0)


def test_export_replays_all_model_turns_with_history_and_no_pseudo_labels():
    records = _records()
    board = BoardState.empty(3)
    for ply, row in enumerate(records):
        replayed = board_from_example(row)
        assert replayed.current_board() == board.current_board()
        assert replayed.moves == board.moves
        assert row["moves"] == _episode()["moves"][:ply]
        assert row["move_number"] == ply and row["split"] == "train"
        assert row["learner"]["run_seed"] == 42 and row["learner"]["update"] == 2
        assert row["learner"]["episode_truncated"] is True
        assert row["learner"]["behavior_temperature"] == .8
        assert row["learner"]["starting_checkpoint"]["path"] == RUN["model"]
        assert not {"response", "result", "source_result", "teacher_value", "policy_target"} & row.keys()
        board.apply_move(board.next_player(), row["learner_move"], strict=True)
    assert len({row["game_id"] for row in records}) == 1


def test_completed_game_keeps_adjudication_as_source_provenance_only():
    records = _records(_episode([["B", "A3"], ["W", "PASS"], ["B", "PASS"]], completed=True))
    assert len(records) == 3
    assert all(row["source_result"] == "W+7.5" and "result" not in row for row in records)
    assert records[-1]["moves"] == [["B", "A3"], ["W", "PASS"]]
    assert board_from_example(records[-1]).consecutive_passes() == 1


@pytest.mark.parametrize("score,outcome", [("B+nan", 1.), ("B+1.5", -1.), ("W+1.5", True)])
def test_completed_game_rejects_invalid_or_inconsistent_adjudication(score, outcome):
    episode = _episode([["B", "PASS"], ["W", "PASS"]], completed=True)
    episode.update(final_score=score, black_outcome=outcome)
    with pytest.raises(ValueError):
        _records(episode)


def test_teacher_annotation_relabels_learner_state_without_losing_behavior_provenance():
    from go_rl.katago import KataGoResult
    from go_rl.teacher_data import make_teacher_record

    source = _records()[1]
    snapshot = copy.deepcopy(source)
    board = board_from_example(source)
    legal = board.legal_moves(superko=True)
    policy = [.1 if action_to_move(index, 3) in legal else -1. for index in range(10)]
    analysis = KataGoResult(.65, {"policy": policy, "moveInfos": [{"move": "C3", "visits": 8}]},
                           score_lead=1.5, best_move="C3")
    labeled = make_teacher_record(source, board, analysis, prior_mix=.5)
    assert source == snapshot and source["teacher_labeled"] is False
    assert labeled["teacher_labeled"] is True and labeled["response"] == "C3"
    for key in ("split", "game_id", "learner", "learner_move", "moves", "initial_stones"):
        assert labeled[key] == source[key]
    assert "result" not in labeled and "source_result" not in labeled


def test_deduplication_removes_equivalent_states_and_heldout_states_without_labels():
    original = _records()
    rotated = [transform_example(row, 1) for row in original]
    heldout = {**original[1], "game_id": "heldout", "response": "UNREAD_LABEL", "split": "validation"}
    kept, counts = deduplicate_training_records([*original, *rotated], excluded_records=[heldout])
    assert len(kept) == 2
    assert counts == {"input_positions": 6, "duplicate_states": 2, "excluded_states": 2,
                      "exported_positions": 2}
    assert all(row["split"] == "train" for row in kept)
    with pytest.raises(ValueError, match="training-only"):
        deduplicate_training_records([heldout])


def test_game_identity_is_rotation_and_run_invariant():
    original = _records()
    final = BoardState.empty(3)
    for player, move in _episode()["moves"]:
        final.apply_move(player, move, strict=True)
    from go_rl.data_prep import state_record_from_board
    rotated = transform_example(state_record_from_board(final), 1)
    other = episode_training_records(_episode(rotated["moves"]), RUN, source_run_id="run-2", episode_row=9)
    assert original[0]["game_id"] == other[0]["game_id"]
    assert original[0]["learner"]["source_run_id"] != other[0]["learner"]["source_run_id"]


@pytest.mark.parametrize("mutate,message", [
    (lambda ep: ep.update(black_outcome=1.), "truncated"),
    (lambda ep: ep.update(plies=7), "plies"),
    (lambda ep: ep["moves"].__setitem__(1, ["B", "B2"]), "alternate"),
    (lambda ep: ep["moves"].__setitem__(1, ["W", "A3"]), "Illegal"),
    (lambda ep: ep.update(behavior_temperature=0), "temperature"),
    (lambda ep: ep.update(terminated=True, truncated=False, black_outcome=1.,
                          final_score="B+1", outcome_source="katago_gtp_final_score"), "two-pass"),
])
def test_corrupted_trajectories_fail_before_export(mutate, message):
    episode = copy.deepcopy(_episode())
    mutate(episode)
    with pytest.raises(ValueError, match=message):
        _records(episode)


def test_export_cli_publishes_manifest_and_refuses_to_overwrite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "export_learner_positions_test", Path(__file__).resolve().parents[1] / "scripts/export_learner_positions.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rl_run.json").write_text(json.dumps(RUN))
    (run_dir / "episodes.jsonl").write_text(json.dumps(_episode()) + "\n")
    output = tmp_path / "positions.jsonl"
    monkeypatch.setattr(sys, "argv", ["export", "--run-dir", str(run_dir), "--output", str(output)])
    runner.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert len(rows) == 3 and manifest["split"] == "train"
    assert manifest["sampled_actions_are_expert_labels"] is False
    assert manifest["sources"][0]["starting_checkpoint"]["available_at_export"] is False
    snapshot = output.read_bytes()
    with pytest.raises(SystemExit):
        runner.main()
    assert output.read_bytes() == snapshot
