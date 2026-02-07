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


def test_parse_match_config_player_name(tmp_path):
    module = _load_eval_module()
    cfg = tmp_path / "match.cfg"
    cfg.write_text(
        """
players = player1,player2
player1 = {
  name = TestModel
  command = python run.py
}
player2 = {
  name = KataGoBaseline
  command = /usr/bin/katago gtp
}
""".strip(),
        encoding="utf-8",
    )
    assert module._parse_match_config_player_name(str(cfg)) == "TestModel"


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
