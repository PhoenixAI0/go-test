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
import os
import subprocess
import sys


def main() -> None:
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
        return

    try:
        subprocess.run(cmd, check=True)
        print(f"\nMatch completed successfully!")
        print(f"  Results logged to: {args.log_file}")
        print(f"  SGF files written to: {args.sgf_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Match failed with exit code {e.returncode}", file=sys.stderr)
        print(f"Check log file for details: {args.log_file}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMatch interrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
