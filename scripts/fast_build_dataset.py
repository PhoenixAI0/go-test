#!/usr/bin/env python3
"""Fast dataset builder with automatic optimization.

This script automatically chooses the fastest available method:
1. Go parser (if available) - fastest for large datasets
2. Python multiprocessing - good for medium datasets
3. Sequential Python - fallback for small datasets

Usage:
    python scripts/fast_build_dataset.py --sgf-dir data/sgf --output data/output.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.data_prep import build_sft_dataset
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv, ensure_go_parser_binary, run_go_parser

logger = logging.getLogger(__name__)


def build_with_go_parser(
    sgf_dir: str,
    output: str,
    max_positions: int,
    num_workers: int,
) -> bool:
    """Build dataset using the Go parser."""
    logger.info("Using Go parser")
    return run_go_parser(
        sgf_dir=sgf_dir,
        output_path=output,
        max_positions=max_positions,
        num_workers=num_workers,
    )


def estimate_dataset_size(sgf_dir: str) -> int:
    """Estimate number of SGF files in directory.

    Args:
        sgf_dir: Directory to scan

    Returns:
        Estimated number of SGF files
    """
    sgf_files = list(Path(sgf_dir).glob("**/*.sgf"))
    return len(sgf_files)


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Fast dataset builder with automatic optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best method
  python scripts/fast_build_dataset.py --sgf-dir data/sgf --output data/sft.jsonl

  # Force Python processing
  python scripts/fast_build_dataset.py --sgf-dir data/sgf --output data/sft.jsonl --no-go-parser

  # Force sequential (single-threaded)
  python scripts/fast_build_dataset.py --sgf-dir data/sgf --output data/sft.jsonl --sequential

Performance:
  - Go parser: Best for 10,000+ SGF files (10-50x faster than Python)
  - Python parallel: Best for 100-10,000 SGF files (5-10x faster than sequential)
  - Python sequential: Best for <100 SGF files
        """,
    )

    parser.add_argument(
        "--sgf-dir",
        required=True,
        help="Directory containing SGF files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
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
        help="Number of workers (0 = auto-detect, default: 0)",
    )
    parser.add_argument(
        "--no-go-parser",
        action="store_true",
        help="Don't use Go parser even if available",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential processing (no parallelism)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for buffered writes (default: 1000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Estimate dataset size
    num_files = estimate_dataset_size(args.sgf_dir)
    logger.info("Found approximately %d SGF files", num_files)

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers == 0:
        import multiprocessing

        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Choose best method
    use_go_parser = False
    use_parallel = True

    if args.sequential:
        use_parallel = False
        logger.info("Using sequential processing (forced)")
    elif args.no_go_parser:
        logger.info("Go parser disabled by user")
    else:
        go_parser = ensure_go_parser_binary()
        if go_parser:
            use_go_parser = True
            logger.info("Defaulting to Go parser")
        elif num_files >= 10:
            logger.info("Auto-selected Python parallel processing")
        else:
            use_parallel = False
            logger.info("Auto-selected sequential processing (small dataset)")

    # Build dataset
    success = False

    if use_go_parser:
        logger.info("Building dataset with Go parser...")
        success = build_with_go_parser(
            args.sgf_dir,
            args.output,
            args.max_positions,
            num_workers,
        )

    if not success:
        if use_go_parser:
            logger.warning("Go parser failed, falling back to Python")

        logger.info("Building dataset with Python...")
        count = build_sft_dataset(
            args.sgf_dir,
            args.output,
            args.max_positions,
            num_workers=num_workers if use_parallel else 1,
            use_parallel=use_parallel,
            batch_size=args.batch_size,
        )
        logger.info("Wrote %d samples to %s", count, args.output)

    logger.info("Dataset building complete!")


if __name__ == "__main__":
    main()
