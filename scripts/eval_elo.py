#!/usr/bin/env python3
"""Evaluate a trained model against KataGo and estimate an Elo difference.

Create a match config with scripts/create_match_config.py, then run:
    uv run python scripts/eval_elo.py --match-config config/my_match.cfg
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import TypedDict

from sgfmill import sgf

from go_rl.gtp_match import MatchConfig, run_match
from go_rl.game_result import parse_game_result
from go_rl.wandb_utils import configure_wandb, finish_wandb, log_training_metrics
from venv_utils import ensure_virtualenv


_CAP_COMMENT = re.compile(r"(?:^|\s)termination=move_cap(?:\s|$)")


class MatchSummary(TypedDict):
    model_name: str
    katago_name: str | None
    total_sgf_files: int
    scanned_sgf_files: int
    stale_sgf_files_ignored: int
    identified_games: int
    wins: int
    losses: int
    draws: int
    skipped_games: int
    void_games: int
    capped_games: int
    cap_rate: float
    win_rate: float | None
    score_rate: float | None
    elo_delta: float | None


def _snapshot_sgf_files(sgf_dir: Path) -> dict[str, tuple[int, int, int]]:
    """Capture SGF file signatures for stale-file filtering."""
    file_stats: dict[str, tuple[int, int, int]] = {}
    for sgf_file in sgf_dir.glob("*.sgf"):
        try:
            stat = sgf_file.stat()
        except OSError:
            continue
        file_stats[str(sgf_file.resolve())] = (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    return file_stats


def _extract_winner_color(result: str) -> str | None:
    try:
        _, black_outcome = parse_game_result(result)
    except ValueError:
        return None
    return "B" if black_outcome > 0 else "W" if black_outcome < 0 else None


def _safe_elo_from_score(score_rate: float) -> float | None:
    # All-win/all-loss samples have no finite maximum-likelihood Elo estimate.
    # Clipping them to an arbitrary epsilon fabricates a precise-looking number.
    if not 0 < score_rate < 1:
        return None
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def summarize_match_results(
    sgf_dir: str,
    model_name: str,
    katago_name: str | None = None,
    baseline_sgf_stats: dict[str, tuple[int, int, int]] | None = None,
) -> MatchSummary:
    sgf_path = Path(sgf_dir)
    all_sgf_files = sorted(sgf_path.glob("*.sgf"))
    if baseline_sgf_stats is None:
        sgf_files = all_sgf_files
        stale_ignored = 0
    else:
        sgf_files = []
        stale_ignored = 0
        for sgf_file in all_sgf_files:
            try:
                stat = sgf_file.stat()
            except OSError:
                continue
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if baseline_sgf_stats.get(str(sgf_file.resolve())) == signature:
                stale_ignored += 1
                continue
            sgf_files.append(sgf_file)

    if not sgf_files:
        if baseline_sgf_stats is None:
            raise RuntimeError(f"No SGF files found in: {sgf_dir}")
        raise RuntimeError(f"No new SGF files found in: {sgf_dir}")

    wins = 0
    losses = 0
    draws = 0
    skipped = 0
    void_games = 0
    capped_games = 0

    for sgf_file in sgf_files:
        try:
            root = sgf.Sgf_game.from_bytes(sgf_file.read_bytes()).get_root()
            black_name = root.get("PB").strip()
            white_name = root.get("PW").strip()
            result = root.get("RE").strip()
            comment = root.get("C") if root.has_property("C") else ""
        except (ValueError, KeyError, UnicodeError):
            skipped += 1
            continue
        winner = _extract_winner_color(result)

        if (black_name == model_name) == (white_name == model_name):
            skipped += 1
            continue
        if black_name == model_name:
            model_color = "B"
            opponent_name = white_name
        else:
            model_color = "W"
            opponent_name = black_name
        if katago_name is not None and opponent_name != katago_name:
            skipped += 1
            continue

        if winner is None:
            normalized = result.upper()
            if normalized in {"0", "DRAW", "JIGO"}:
                draws += 1
            elif normalized in {"VOID", "?"}:
                void_games += 1
                if _CAP_COMMENT.search(comment):
                    capped_games += 1
            else:
                skipped += 1
            continue
        if winner == model_color:
            wins += 1
        else:
            losses += 1

    identified_games = wins + losses + draws
    if identified_games + void_games == 0:
        raise RuntimeError(
            "Could not identify any games containing the configured model name in SGFs."
        )

    win_rate = wins / identified_games if identified_games else None
    score_rate = (wins + 0.5 * draws) / identified_games if identified_games else None
    summary = {
        "model_name": model_name,
        "katago_name": katago_name,
        "total_sgf_files": len(sgf_files),
        "scanned_sgf_files": len(all_sgf_files),
        "stale_sgf_files_ignored": stale_ignored,
        "identified_games": identified_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "skipped_games": skipped,
        "void_games": void_games,
        "capped_games": capped_games,
        "cap_rate": capped_games / (identified_games + void_games),
        "win_rate": win_rate,
        "score_rate": score_rate,
        "elo_delta": _safe_elo_from_score(score_rate) if score_rate is not None else None,
    }
    return summary


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-config", required=True,
                        help="Match config from scripts/create_match_config.py")
    parser.add_argument("--log-file", default="match.log",
                        help="Match log, including engine stderr (default: match.log)")
    parser.add_argument("--sgf-dir", help="SGF output directory (default: from the match config)")
    parser.add_argument("--dry-run", action="store_true", help="Print the match plan without playing")
    parser.add_argument("--summary-json", help="Optional path to write JSON summary metrics.")
    parser.add_argument("--model-name", help="Override the model name from the match config.")
    parser.add_argument("--katago-name", help="Override the KataGo name from the match config.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases reporting for evaluation metrics.")
    parser.add_argument("--wandb-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        help="Weights & Biases mode.")
    parser.add_argument("--wandb-prefix", default="eval",
                        help="Metric prefix for W&B logs (default: eval).")
    parser.add_argument("--wandb-run-name", help="Optional custom W&B run name for evaluation.")
    args = parser.parse_args()

    try:
        config = MatchConfig.load(args.match_config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: Invalid match config {args.match_config}: {exc}", file=sys.stderr)
        print("Create one with scripts/create_match_config.py.", file=sys.stderr)
        sys.exit(1)
    if args.model_name:
        config.model.name = args.model_name
    if args.katago_name:
        config.katago.name = args.katago_name
    sgf_dir = Path(args.sgf_dir or config.sgf_output_dir)
    log_file = Path(args.log_file)

    print(f"Match: {config.model.name} vs {config.katago.name}, {config.num_games} games "
          f"on {config.board_size}x{config.board_size}")
    print(f"  Model: {config.model.command}")
    print(f"  KataGo: {config.katago.command}")
    print(f"  Log: {log_file}")
    print(f"  SGF output: {sgf_dir}/")
    if args.dry_run:
        print("Dry run - not playing")
        return

    run_name = args.wandb_run_name or f"eval_{Path(args.match_config).stem}"
    wandb_enabled = configure_wandb(
        args.wandb,
        run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            "match_config": args.match_config,
            "sgf_dir": str(sgf_dir),
            "model_name": config.model.name,
            "katago_name": config.katago.name,
        },
        tags=["eval", "katago-match"],
    )
    sgf_dir.mkdir(parents=True, exist_ok=True)
    pre_match_sgf_stats = _snapshot_sgf_files(sgf_dir)
    try:
        run_match(config, sgf_dir, log_file)
        summary = summarize_match_results(
            sgf_dir=str(sgf_dir),
            model_name=config.model.name,
            katago_name=config.katago.name,
            baseline_sgf_stats=pre_match_sgf_stats,
        )
        print(
            "Summary: "
            f"win_rate={summary['win_rate']}, "
            f"score_rate={summary['score_rate']}, "
            f"elo_delta={summary['elo_delta']}, "
            f"wins={summary['wins']}, "
            f"losses={summary['losses']}, "
            f"draws={summary['draws']}, "
            f"capped_games={summary['capped_games']}"
        )

        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"JSON summary written to: {summary_path}")

        if wandb_enabled:
            prefix = args.wandb_prefix.rstrip("/")
            metrics = {
                f"{prefix}/win_rate": summary["win_rate"],
                f"{prefix}/score_rate": summary["score_rate"],
                f"{prefix}/elo_delta": summary["elo_delta"],
                f"{prefix}/wins": summary["wins"],
                f"{prefix}/losses": summary["losses"],
                f"{prefix}/draws": summary["draws"],
                f"{prefix}/identified_games": summary["identified_games"],
                f"{prefix}/void_games": summary["void_games"],
                f"{prefix}/capped_games": summary["capped_games"],
                f"{prefix}/cap_rate": summary["cap_rate"],
                f"{prefix}/total_sgf_files": summary["total_sgf_files"],
                f"{prefix}/skipped_games": summary["skipped_games"],
                f"{prefix}/stale_sgf_files_ignored": summary["stale_sgf_files_ignored"],
            }
            log_training_metrics(step=0, metrics={key: value for key, value in metrics.items() if value is not None})
    except (OSError, RuntimeError) as exc:
        print(f"\nError: Match failed: {exc}", file=sys.stderr)
        print(f"Check the log file for details: {log_file}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMatch interrupted by user", file=sys.stderr)
        sys.exit(130)
    finally:
        if wandb_enabled:
            finish_wandb()


if __name__ == "__main__":
    main()
