#!/usr/bin/env python3
"""Generate a match configuration for eval_elo.py.

The config names a GTP command for the trained model and one for the KataGo
baseline; eval_elo.py referees games between them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

from go_rl.gtp_match import MatchConfig, PlayerConfig
from venv_utils import ensure_virtualenv

DEFAULT_GTP_CONFIG = Path(__file__).resolve().parents[1] / "config" / "katago_gtp_eval.cfg"


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def create_match_config(
    output_path: str,
    board_size: int,
    num_games: int,
    model_command: str,
    katago_bin: str,
    katago_model: str,
    katago_config: str = str(DEFAULT_GTP_CONFIG),
    sgf_output_dir: str = "match_sgfs",
    main_time: int = 60,
    byoyomi_time: int = 5,
    byoyomi_periods: int = 5,
    komi: float = 7.5,
    rules: str = "chinese",
    model_name: str = "TrainedModel",
    katago_name: str = "KataGoBaseline",
) -> None:
    """Write a JSON match config; `katago_config` must be a KataGo GTP config."""
    katago_paths = [_expand(path) for path in (katago_bin, katago_model, katago_config)]
    for path in katago_paths:
        if "$" in path:
            raise ValueError(f"Unresolved environment variable in KataGo path: {path}")
    katago_bin, katago_model, katago_config = katago_paths
    config = MatchConfig(
        board_size=board_size,
        num_games=num_games,
        model=PlayerConfig(model_name, _expand(model_command)),
        katago=PlayerConfig(katago_name, shlex.join(
            [katago_bin, "gtp", "-model", katago_model, "-config", katago_config])),
        komi=komi,
        rules=rules,
        main_time=main_time,
        byoyomi_time=byoyomi_time,
        byoyomi_periods=byoyomi_periods,
        sgf_output_dir=sgf_output_dir,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    print(f"Match config written to: {output_path}")
    print(f"  Board size: {board_size}x{board_size}")
    print(f"  Number of games: {num_games}")
    print(f"  SGF output: {sgf_output_dir}/")
    print("\nTo run the match:")
    print(f"  uv run python scripts/eval_elo.py --match-config {output_path}")


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Generate KataGo match configuration for Elo evaluation"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write match config file",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=9,
        choices=[9, 19],
        help="Board size (9 or 19, default: 9)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=200,
        help="Number of games to play (default: 200)",
    )
    parser.add_argument(
        "--model-command",
        required=True,
        help="Command to run your trained model as GTP engine",
    )
    parser.add_argument(
        "--katago-bin",
        default="${KATAGO_BIN}",
        help="Path to KataGo binary (default: ${KATAGO_BIN})",
    )
    parser.add_argument(
        "--katago-model",
        default="${KATAGO_MODEL}",
        help="Path to KataGo model (default: ${KATAGO_MODEL})",
    )
    parser.add_argument(
        "--katago-config",
        default=str(DEFAULT_GTP_CONFIG),
        help="KataGo GTP config (default: config/katago_gtp_eval.cfg)",
    )
    parser.add_argument(
        "--sgf-output-dir",
        default="match_sgfs",
        help="Directory for SGF output (default: match_sgfs)",
    )
    parser.add_argument(
        "--main-time",
        type=int,
        default=None,
        help="Main time in seconds (default: 60 for 9x9, 300 for 19x19)",
    )
    parser.add_argument(
        "--byoyomi-time",
        type=int,
        default=None,
        help="Byo-yomi time per move (default: 5 for 9x9, 10 for 19x19)",
    )
    parser.add_argument(
        "--byoyomi-periods",
        type=int,
        default=5,
        help="Number of byo-yomi periods (default: 5)",
    )
    parser.add_argument(
        "--komi",
        type=float,
        default=7.5,
        help="Komi value (default: 7.5)",
    )
    parser.add_argument(
        "--rules",
        default="chinese",
        help="Rule set to use (default: chinese)",
    )
    parser.add_argument(
        "--model-name",
        default="TrainedModel",
        help="Display name for trained model (default: TrainedModel)",
    )
    parser.add_argument(
        "--katago-name",
        default="KataGoBaseline",
        help="Display name for KataGo (default: KataGoBaseline)",
    )

    args = parser.parse_args()

    # Set default time controls based on board size
    if args.main_time is None:
        args.main_time = 60 if args.board_size == 9 else 300
    if args.byoyomi_time is None:
        args.byoyomi_time = 5 if args.board_size == 9 else 10

    try:
        create_match_config(
            output_path=args.output,
            board_size=args.board_size,
            num_games=args.num_games,
            model_command=args.model_command,
            katago_bin=args.katago_bin,
            katago_model=args.katago_model,
            katago_config=args.katago_config,
            sgf_output_dir=args.sgf_output_dir,
            main_time=args.main_time,
            byoyomi_time=args.byoyomi_time,
            byoyomi_periods=args.byoyomi_periods,
            komi=args.komi,
            rules=args.rules,
            model_name=args.model_name,
            katago_name=args.katago_name,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
