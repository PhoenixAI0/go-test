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
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.data_prep import build_sft_dataset
from go_rl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def find_go_parser() -> Path | None:
    """Find the Go SGF parser binary if available.

    Returns:
        Path to binary or None if not found
    """
    # Check in go_parser directory
    repo_root = Path(__file__).parent.parent
    go_parser_path = repo_root / "go_parser" / "sgf_parser"

    if go_parser_path.exists() and os.access(go_parser_path, os.X_OK):
        return go_parser_path

    # Check in PATH
    go_parser_in_path = shutil.which("sgf_parser")
    if go_parser_in_path:
        return Path(go_parser_in_path)

    return None


def build_with_go_parser(
    sgf_dir: str,
    output: str,
    max_positions: int,
    num_workers: int,
) -> bool:
    """Build dataset using the Go parser.

    Args:
        sgf_dir: Directory containing SGF files
        output: Output file path
        max_positions: Maximum positions to collect
        num_workers: Number of workers

    Returns:
        True if successful, False otherwise
    """
    go_parser = find_go_parser()
    if not go_parser:
        logger.warning("Go parser not found")
        return False

    logger.info("Using Go parser: %s", go_parser)

    cmd = [
        str(go_parser),
        "-sgf-dir",
        sgf_dir,
        "-output",
        output,
        "-max-positions",
        str(max_positions),
        "-workers",
        str(num_workers),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Go parser output:\n%s", result.stdout)
        if result.stderr:
            logger.warning("Go parser stderr:\n%s", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Go parser failed: %s", e)
        logger.error("Stdout: %s", e.stdout)
        logger.error("Stderr: %s", e.stderr)
        return False
    except Exception as e:
        logger.error("Error running Go parser: %s", e)
        return False


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
        # Auto-detect best method
        go_parser = find_go_parser()
        if go_parser and num_files >= 1000:
            use_go_parser = True
            logger.info("Auto-selected Go parser (large dataset)")
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
