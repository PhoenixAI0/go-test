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
