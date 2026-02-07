#!/usr/bin/env python3
"""Build SFT datasets using the Go SGF parser."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.data_prep import build_sft_dataset
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


def estimate_dataset_size(sgf_dir: str) -> int:
    return len(list(Path(sgf_dir).glob("**/*.sgf")))


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(description="Go-parser dataset builder")
    parser.add_argument("--sgf-dir", required=True, help="Directory containing SGF files")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=200000,
        help="Maximum positions to collect (default: 200000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Go parser worker count (0 = parser default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    num_files = estimate_dataset_size(args.sgf_dir)
    logger.info("Found approximately %d SGF files", num_files)
    count = build_sft_dataset(
        sgf_dir=args.sgf_dir,
        output_path=args.output,
        max_positions=args.max_positions,
        num_workers=args.num_workers,
    )
    logger.info("Wrote %d samples to %s", count, args.output)


if __name__ == "__main__":
    main()
