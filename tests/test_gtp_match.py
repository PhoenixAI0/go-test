from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from go_rl.gtp_match import GtpEngine, MatchConfig, PlayerConfig, _Clock, run_match

ROOT = Path(__file__).resolve().parents[1]

FAKE_ENGINE = r'''
import argparse, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--moves", default="")
parser.add_argument("--final-score", default="0")
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--reject-play", action="store_true")
parser.add_argument("--log")
args = parser.parse_args()
moves = iter([move for move in args.moves.split(",") if move])
for line in sys.stdin:
    command = line.strip()
    if args.log:
        with open(args.log, "a") as handle:
            handle.write(command + "\n")
    name = command.split()[0] if command else ""
    if name == "quit":
        print("=\n", flush=True)
        break
    if name == "genmove":
        time.sleep(args.sleep)
        print(f"= {next(moves, 'pass')}\n", flush=True)
    elif name == "play" and args.reject_play:
        print("? illegal move\n", flush=True)
    elif name == "final_score":
        print(f"= {args.final_score}\n", flush=True)
    elif name.startswith("kata") or name.startswith("kgs") or name == "time_left":
        print("? unknown command\n", flush=True)
    else:
        print("=\n", flush=True)
'''


@pytest.fixture
def engine(tmp_path):
    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE_ENGINE)

    def command(*flags: str) -> str:
        return shlex.join([sys.executable, str(script), *flags])

    return command


def _config(model_command: str, katago_command: str, **overrides) -> MatchConfig:
    values = dict(board_size=9, num_games=1, model=PlayerConfig("Model", model_command),
                  katago=PlayerConfig("KataGo", katago_command))
    values.update(overrides)
    return MatchConfig(**values)


def _play(tmp_path, config: MatchConfig) -> list[str]:
    paths = run_match(config, tmp_path / "sgf", tmp_path / "match.log")
    return [path.read_text() for path in paths]


def test_match_alternates_colors_scores_with_katago_and_writes_sgfs(tmp_path, engine):
    katago_log = tmp_path / "katago.log"
    config = _config(engine("--moves", "E5"),
                     engine("--final-score", "W+7.5", "--log", str(katago_log)), num_games=2)
    first, second = _play(tmp_path, config)

    # The model opens as Black, then both engines pass; KataGo scores the game.
    assert first == ("(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RU[chinese]PB[Model]PW[KataGo]"
                     "RE[W+7.5];B[ee];W[];B[])\n")
    assert "PB[KataGo]PW[Model]RE[W+7.5];B[];W[])" in second
    commands = katago_log.read_text().splitlines()
    assert "play b E5" in commands and "kata-set-rules chinese" in commands

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from eval_elo import summarize_match_results
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    summary = summarize_match_results(str(tmp_path / "sgf"), "Model")
    assert (summary["wins"], summary["losses"]) == (1, 1)


@pytest.mark.parametrize("moves,expected", [
    ("E5,E5", "W+F"),  # Replaying on an occupied point.
    ("Z99", "W+F"),  # Off the board.
    ("resign", "W+R"),
])
def test_model_loses_by_illegal_move_or_resignation(tmp_path, engine, moves, expected):
    (sgf,) = _play(tmp_path, _config(engine("--moves", moves), engine()))
    assert f"RE[{expected}]" in sgf


def test_model_loses_on_time_after_byoyomi(tmp_path, engine):
    config = _config(engine("--sleep", "0.3"), engine(),
                     main_time=0, byoyomi_time=0.1, byoyomi_periods=1)
    (sgf,) = _play(tmp_path, config)
    assert "RE[W+T]" in sgf


def test_katago_rejection_forfeits_but_model_rejection_is_an_engine_error(tmp_path, engine):
    # KataGo is trusted: if it refuses the model's move, the model forfeits.
    (sgf,) = _play(tmp_path, _config(engine("--moves", "E5"), engine("--reject-play")))
    assert "RE[W+F]" in sgf

    # A model engine refusing a legal KataGo move must not hand the model a win.
    with pytest.raises(RuntimeError, match="Model rejected legal move W E5"):
        _play(tmp_path, _config(engine("--reject-play"), engine("--moves", "E5")))


def test_move_cap_ends_game_with_katago_score(tmp_path, engine):
    config = _config(engine("--moves", "E5,D4"), engine("--moves", "C3,F6", "--final-score", "B+3"),
                     max_moves=3)
    (sgf,) = _play(tmp_path, config)
    assert sgf.endswith("RE[B+3];B[ee];W[cg];B[df])\n")


def test_crashed_engine_is_reported(tmp_path):
    engine = GtpEngine("Broken", shlex.join([sys.executable, "-c", "pass"]))
    with pytest.raises(RuntimeError, match="Broken exited unexpectedly"):
        engine.send("name", timeout=10)
    engine.close()


def test_byoyomi_clock():
    clock = _Clock(main=10, byoyomi=5, periods=2)
    assert clock.spend(9) and clock.main == 1
    assert clock.spend(5) and clock.periods == 2  # 1s main, then within one period.
    assert clock.spend(9) and clock.periods == 1  # Overran one period.
    assert not clock.spend(6)


def test_config_validation():
    with pytest.raises(ValueError, match="distinct names"):
        MatchConfig(9, 1, PlayerConfig("A", "x"), PlayerConfig("A", "y"))
    with pytest.raises(ValueError, match="thinking time"):
        MatchConfig(9, 1, PlayerConfig("A", "x"), PlayerConfig("B", "y"),
                    main_time=0, byoyomi_time=0)


def test_eval_elo_cli_runs_match_and_writes_summary(tmp_path, engine):
    config_path = tmp_path / "match.cfg"
    config = _config(engine("--moves", "E5"), engine("--final-score", "B+1"),
                     sgf_output_dir=str(tmp_path / "sgf"), num_games=2)
    config_path.write_text(json.dumps(asdict(config)))
    summary_path = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_elo.py"), "--match-config", str(config_path),
         "--log-file", str(tmp_path / "match.log"), "--summary-json", str(summary_path)],
        capture_output=True, text=True, env={**os.environ, "VIRTUAL_ENV": sys.prefix},
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text())
    assert (summary["wins"], summary["losses"], summary["identified_games"]) == (1, 1, 2)


@pytest.mark.katago
def test_match_against_real_katago(tmp_path):
    binary, model = os.environ.get("KATAGO_BIN"), os.environ.get("KATAGO_MODEL")
    if not binary or not model:
        pytest.skip("Set KATAGO_BIN and KATAGO_MODEL (source .env)")
    katago = shlex.join([binary, "gtp", "-model", model, "-config",
                         str(ROOT / "config/katago_gtp_eval.cfg"), "-override-config", "maxVisits=2"])
    config = _config(katago, katago, num_games=2)
    games = _play(tmp_path, config)
    assert len(games) == 2
    for game in games:
        result = game.split("RE[")[1].split("]")[0]
        # Both sides are strong engines: games end by scoring, never by forfeit.
        assert result[:2] in {"B+", "W+"} and result[2:] not in {"F", "T"}, game
