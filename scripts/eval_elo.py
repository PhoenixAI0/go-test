#!/usr/bin/env python3
"""Run KataGo match for Elo evaluation.

This script runs a match between a trained model and KataGo to evaluate
the model's playing strength via Elo rating.

Before running this script:
1. Create a match config using scripts/create_match_config.py, OR
2. Manually edit one of the templates in config/katago_match_*.cfg

Example usage:
    # Using a pre-generated match config
    python scripts/eval_elo.py \\
        --katago-bin /path/to/katago \\
        --match-config config/my_match.cfg

    # Using template (requires manual editing first)
    python scripts/eval_elo.py \\
        --katago-bin /path/to/katago \\
        --match-config config/katago_match_9x9.cfg \\
        --log-file logs/match_9x9.log \\
        --sgf-dir match_sgfs/9x9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from go_rl.wandb_utils import configure_wandb, finish_wandb, log_training_metrics
from venv_utils import ensure_virtualenv


_RESULT_PATTERN = re.compile(r"RE\[([^\]]*)\]")
_BLACK_PATTERN = re.compile(r"PB\[([^\]]*)\]")
_WHITE_PATTERN = re.compile(r"PW\[([^\]]*)\]")


def _snapshot_sgf_files(sgf_dir: Path) -> Dict[str, tuple[int, int, int]]:
    """Capture SGF file signatures for stale-file filtering."""
    file_stats: Dict[str, tuple[int, int, int]] = {}
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


def _parse_match_config_player_name(match_config_path: str) -> Optional[str]:
    """Extract player1 name from a KataGo match config."""
    try:
        content = Path(match_config_path).read_text(encoding="utf-8")
    except Exception:
        return None
    match = re.search(
        r"player1\s*=\s*\{[^}]*?\bname\s*=\s*([^\n\r]+)",
        content,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.strip() or None


def _extract_winner_color(result: str) -> Optional[str]:
    normalized = result.strip().upper()
    if normalized.startswith("B+"):
        return "B"
    if normalized.startswith("W+"):
        return "W"
    return None


def _safe_elo_from_score(score_rate: float) -> float:
    clipped = min(max(score_rate, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def summarize_match_results(
    sgf_dir: str,
    model_name: str,
    katago_name: Optional[str] = None,
    baseline_sgf_stats: Optional[Dict[str, tuple[int, int, int]]] = None,
) -> Dict[str, Any]:
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

    for sgf_file in sgf_files:
        content = sgf_file.read_text(encoding="utf-8", errors="ignore")
        black_match = _BLACK_PATTERN.search(content)
        white_match = _WHITE_PATTERN.search(content)
        result_match = _RESULT_PATTERN.search(content)
        if not black_match or not white_match or not result_match:
            skipped += 1
            continue

        black_name = black_match.group(1).strip()
        white_name = white_match.group(1).strip()
        result = result_match.group(1).strip()
        winner = _extract_winner_color(result)

        if black_name == model_name:
            model_color = "B"
        elif white_name == model_name:
            model_color = "W"
        else:
            skipped += 1
            continue

        if winner is None:
            draws += 1
            continue
        if winner == model_color:
            wins += 1
        else:
            losses += 1

    identified_games = wins + losses + draws
    if identified_games == 0:
        raise RuntimeError(
            "Could not identify any games containing the configured model name in SGFs."
        )

    win_rate = wins / identified_games
    score_rate = (wins + 0.5 * draws) / identified_games
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
        "win_rate": win_rate,
        "score_rate": score_rate,
        "elo_delta": _safe_elo_from_score(score_rate),
    }
    return summary


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Run KataGo match for Elo evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run match with custom config
  python scripts/eval_elo.py \\
      --katago-bin /usr/local/bin/katago \\
      --match-config config/my_match.cfg

  # Run match with template (edit template first!)
  python scripts/eval_elo.py \\
      --katago-bin $KATAGO_BIN \\
      --match-config config/katago_match_9x9.cfg

For more information, see docs/EVALUATION_GUIDE.md
        """,
    )
    parser.add_argument(
        "--katago-bin",
        required=True,
        help="Path to KataGo binary",
    )
    parser.add_argument(
        "--match-config",
        required=True,
        help="Path to match configuration file",
    )
    parser.add_argument(
        "--log-file",
        default="match.log",
        help="Path to write match log (default: match.log)",
    )
    parser.add_argument(
        "--sgf-dir",
        default="match_sgfs",
        help="Directory to write SGF files (default: match_sgfs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write JSON summary metrics.",
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Model player name for SGF result parsing. "
            "Defaults to player1 name from match config."
        ),
    )
    parser.add_argument(
        "--katago-name",
        help="Optional KataGo player name for metadata only.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases reporting for evaluation metrics.",
    )
    parser.add_argument("--wandb-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    parser.add_argument(
        "--wandb-prefix",
        default="eval",
        help="Metric prefix for W&B logs (default: eval).",
    )
    parser.add_argument(
        "--wandb-run-name",
        help="Optional custom W&B run name for evaluation.",
    )
    args = parser.parse_args()

    # Validate inputs
    katago_bin = os.path.expanduser(os.path.expandvars(args.katago_bin))
    match_config = os.path.expanduser(os.path.expandvars(args.match_config))

    if not os.path.exists(katago_bin):
        print(f"Error: KataGo binary not found: {katago_bin}", file=sys.stderr)
        print("Set KATAGO_BIN environment variable or provide --katago-bin", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(match_config):
        print(f"Error: Match config not found: {match_config}", file=sys.stderr)
        print("\nTo create a match config:", file=sys.stderr)
        print("  1. Use scripts/create_match_config.py, OR", file=sys.stderr)
        print("  2. Edit a template from config/katago_match_*.cfg", file=sys.stderr)
        sys.exit(1)

    # Check for placeholder paths in config
    with open(match_config, "r", encoding="utf-8") as f:
        config_content = f.read()
        if "/path/to/" in config_content or "${" in config_content:
            print(f"Warning: Match config contains placeholder paths: {match_config}", file=sys.stderr)
            print("Please edit the config and replace placeholders with actual paths", file=sys.stderr)
            print("", file=sys.stderr)

    # Create output directories
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(args.sgf_dir):
        os.makedirs(args.sgf_dir, exist_ok=True)
    pre_match_sgf_stats = _snapshot_sgf_files(Path(args.sgf_dir))

    model_name = args.model_name or _parse_match_config_player_name(match_config)
    if not model_name:
        print(
            "Error: Could not infer model name from match config. Pass --model-name.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = args.wandb_run_name or f"eval_{Path(match_config).stem}"
    wandb_enabled = configure_wandb(
        args.wandb,
        run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            "match_config": match_config,
            "sgf_dir": args.sgf_dir,
            "model_name": model_name,
            "katago_name": args.katago_name,
        },
        tags=["eval", "katago-match"],
    )

    # Build command
    cmd = [
        katago_bin,
        "match",
        "-config",
        match_config,
        "-log-file",
        args.log_file,
        "-sgf-output-dir",
        args.sgf_dir,
    ]

    print(f"Running KataGo match:")
    print(f"  Binary: {katago_bin}")
    print(f"  Config: {match_config}")
    print(f"  Log: {args.log_file}")
    print(f"  SGF output: {args.sgf_dir}/")
    print(f"  Command: {' '.join(cmd)}")
    print()

    if args.dry_run:
        print("Dry run - not executing")
        if wandb_enabled:
            finish_wandb()
        return

    try:
        subprocess.run(cmd, check=True)
        print(f"\nMatch completed successfully!")
        print(f"  Results logged to: {args.log_file}")
        print(f"  SGF files written to: {args.sgf_dir}/")
        summary = summarize_match_results(
            sgf_dir=args.sgf_dir,
            model_name=model_name,
            katago_name=args.katago_name,
            baseline_sgf_stats=pre_match_sgf_stats,
        )
        print(
            "  Summary: "
            f"win_rate={summary['win_rate']:.4f}, "
            f"score_rate={summary['score_rate']:.4f}, "
            f"elo_delta={summary['elo_delta']:.2f}, "
            f"wins={summary['wins']}, "
            f"losses={summary['losses']}, "
            f"draws={summary['draws']}"
        )

        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"  JSON summary written to: {summary_path}")

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
                f"{prefix}/total_sgf_files": summary["total_sgf_files"],
                f"{prefix}/skipped_games": summary["skipped_games"],
                f"{prefix}/stale_sgf_files_ignored": summary["stale_sgf_files_ignored"],
            }
            log_training_metrics(step=0, metrics=metrics)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Match failed with exit code {e.returncode}", file=sys.stderr)
        print(f"Check log file for details: {args.log_file}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMatch interrupted by user", file=sys.stderr)
        sys.exit(130)
    finally:
        if wandb_enabled:
            finish_wandb()


if __name__ == "__main__":
    main()
