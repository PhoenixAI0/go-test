import importlib.util
import sys
from pathlib import Path

import pytest


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "eval_elo.py"
    spec = importlib.util.spec_from_file_location("eval_elo_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load eval_elo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_match_config_round_trips_and_quotes_paths(tmp_path, monkeypatch):
    import shlex

    _load_eval_module()  # Puts scripts/ on sys.path.
    from create_match_config import create_match_config
    from go_rl.gtp_match import MatchConfig

    katago_dir = tmp_path / "kata go"
    output = tmp_path / "match.cfg"
    monkeypatch.setenv("KATAGO_BIN", str(katago_dir / "katago"))
    create_match_config(str(output), board_size=19, num_games=4, model_command="engine --x 1",
                        katago_bin="${KATAGO_BIN}", katago_model=str(katago_dir / "m.bin.gz"),
                        model_name="TestModel")
    config = MatchConfig.load(output)
    assert (config.board_size, config.num_games, config.model.name) == (19, 4, "TestModel")
    assert config.max_moves == 3 * 19 * 19
    command = shlex.split(config.katago.command)
    assert command[:2] == [str(katago_dir / "katago"), "gtp"]
    assert command[command.index("-config") + 1].endswith("config/katago_gtp_eval.cfg")

    monkeypatch.delenv("KATAGO_BIN")
    with pytest.raises(ValueError, match="Unresolved environment variable"):
        create_match_config(str(output), 9, 1, "engine", "${KATAGO_BIN}", "model")


def test_summarize_match_results(tmp_path):
    module = _load_eval_module()
    sgf_dir = tmp_path / "sgfs"
    sgf_dir.mkdir()

    # Win (model is black)
    (sgf_dir / "g1.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[TestModel]PW[KataGoBaseline]RE[B+R])",
        encoding="utf-8",
    )
    # Loss (model is white)
    (sgf_dir / "g2.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[KataGoBaseline]PW[TestModel]RE[B+2.5])",
        encoding="utf-8",
    )
    # Draw
    (sgf_dir / "g3.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[TestModel]PW[KataGoBaseline]RE[0])",
        encoding="utf-8",
    )
    # Skipped (different players)
    (sgf_dir / "g4.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[A]PW[B]RE[W+R])",
        encoding="utf-8",
    )

    summary = module.summarize_match_results(
        sgf_dir=str(sgf_dir),
        model_name="TestModel",
        katago_name="KataGoBaseline",
    )
    assert summary["identified_games"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["skipped_games"] == 1
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["score_rate"] == pytest.approx(0.5)
    assert summary["elo_delta"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_match_results_requires_sgfs(tmp_path):
    module = _load_eval_module()
    sgf_dir = tmp_path / "empty"
    sgf_dir.mkdir()
    with pytest.raises(RuntimeError, match="No SGF files"):
        module.summarize_match_results(sgf_dir=str(sgf_dir), model_name="TestModel")


def test_summarize_match_results_ignores_stale_sgfs(tmp_path):
    module = _load_eval_module()
    sgf_dir = tmp_path / "sgfs"
    sgf_dir.mkdir()

    (sgf_dir / "old.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[TestModel]PW[KataGoBaseline]RE[W+R])",
        encoding="utf-8",
    )
    baseline = module._snapshot_sgf_files(sgf_dir)

    (sgf_dir / "new.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[TestModel]PW[KataGoBaseline]RE[B+R])",
        encoding="utf-8",
    )

    summary = module.summarize_match_results(
        sgf_dir=str(sgf_dir),
        model_name="TestModel",
        baseline_sgf_stats=baseline,
    )
    assert summary["total_sgf_files"] == 1
    assert summary["scanned_sgf_files"] == 2
    assert summary["stale_sgf_files_ignored"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0


def test_summarize_match_results_requires_new_sgfs_with_baseline(tmp_path):
    module = _load_eval_module()
    sgf_dir = tmp_path / "sgfs"
    sgf_dir.mkdir()
    (sgf_dir / "old.sgf").write_text(
        "(;GM[1]FF[4]SZ[9]PB[TestModel]PW[KataGoBaseline]RE[B+R])",
        encoding="utf-8",
    )
    baseline = module._snapshot_sgf_files(sgf_dir)

    with pytest.raises(RuntimeError, match="No new SGF files"):
        module.summarize_match_results(
            sgf_dir=str(sgf_dir),
            model_name="TestModel",
            baseline_sgf_stats=baseline,
        )


def test_void_and_unknown_games_are_not_draws(tmp_path):
    module = _load_eval_module()
    for index, result in enumerate(["B+R", "Void", "?", "garbage", "0"]):
        comment = "C[termination=move_cap]" if result == "Void" else ""
        (tmp_path / f"{index}.sgf").write_text(
            f"(;PB[Model]PW[KataGo]RE[{result}]{comment})")
    summary = module.summarize_match_results(str(tmp_path), "Model")
    assert summary["identified_games"] == 2
    assert summary["wins"] == 1 and summary["draws"] == 1
    assert summary["void_games"] == 2 and summary["skipped_games"] == 1
    assert summary["capped_games"] == 1 and summary["cap_rate"] == 0.25
    assert summary["score_rate"] == 0.75


def test_all_capped_games_report_missing_strength_metrics(tmp_path):
    module = _load_eval_module()
    (tmp_path / "capped.sgf").write_text(
        "(;PB[Model]PW[KataGo]RE[Void]C[termination=move_cap])")
    summary = module.summarize_match_results(str(tmp_path), "Model")
    assert summary["identified_games"] == 0
    assert summary["capped_games"] == 1 and summary["cap_rate"] == 1.0
    assert summary["win_rate"] is None and summary["elo_delta"] is None


def test_malformed_winning_prefix_does_not_become_a_win(tmp_path):
    module = _load_eval_module()
    for index, result in enumerate(["B+nan", "W+-1", "B+", "B+garbage", "W+1.5"]):
        (tmp_path / f"{index}.sgf").write_text(f"(;PB[Model]PW[KataGo]RE[{result}])")
    summary = module.summarize_match_results(str(tmp_path), "Model")
    assert summary["identified_games"] == 1
    assert summary["wins"] == 0 and summary["losses"] == 1
    assert summary["skipped_games"] == 4
    assert summary["elo_delta"] is None  # One loss cannot give a finite Elo MLE.


def test_match_summary_parses_escaped_names_and_does_not_mix_opponents(tmp_path):
    module = _load_eval_module()
    from go_rl.gtp_match import GameRecord, MatchConfig, PlayerConfig, game_to_sgf

    model_name = "Qwen ] \\ model"
    config = MatchConfig(19, 1, PlayerConfig(model_name, "unused"), PlayerConfig("KataGo", "unused"))
    for index, (opponent, result) in enumerate([("KataGo", "W+1.5"), ("OtherEngine", "B+R")]):
        record = GameRecord(model_name, opponent, [], result)
        (tmp_path / f"{index}.sgf").write_text(game_to_sgf(record, config))
    summary = module.summarize_match_results(str(tmp_path), model_name, katago_name="KataGo")
    assert summary["identified_games"] == 1
    assert summary["losses"] == 1 and summary["wins"] == 0 and summary["skipped_games"] == 1
