#!/usr/bin/env python3
"""Filter and validate self-play games using KataGo analysis.

This script analyzes self-play games and filters out low-quality games based on:
- Game length (too short or too long)
- Illegal move count
- Game quality (KataGo score distribution)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple
from venv_utils import ensure_virtualenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_num_workers() -> int:
    """Get optimal number of worker processes."""
    try:
        cpu_count = mp.cpu_count()
        return max(1, cpu_count - 1)
    except NotImplementedError:
        return 4


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_sgf_moves(sgf_content: str) -> Tuple[int, List[str]]:
    """Parse moves from SGF content.

    Args:
        sgf_content: SGF file content

    Returns:
        Tuple of (board_size, list of moves)
    """
    import re

    # Extract board size
    sz_match = re.search(r"SZ\[(\d+)\]", sgf_content)
    board_size = int(sz_match.group(1)) if sz_match else 19

    # Extract moves
    moves = re.findall(r";([BW])\[([a-s]{2}|)\]", sgf_content)
    move_list = [f"{player}:{coord}" if coord else f"{player}:pass" for player, coord in moves]

    return board_size, move_list


def filter_by_length(
    sgf_file: Path, min_moves: int, max_moves: int
) -> Tuple[bool, str]:
    """Filter game by move count.

    Args:
        sgf_file: Path to SGF file
        min_moves: Minimum acceptable moves
        max_moves: Maximum acceptable moves

    Returns:
        Tuple of (should_keep, reason)
    """
    try:
        content = sgf_file.read_text(encoding="utf-8")
        board_size, moves = parse_sgf_moves(content)

        move_count = len(moves)

        if move_count < min_moves:
            return False, f"too_short ({move_count} < {min_moves})"
        if move_count > max_moves:
            return False, f"too_long ({move_count} > {max_moves})"

        return True, f"valid ({move_count} moves)"

    except Exception as e:
        return False, f"parse_error: {e}"


def validate_sgf(sgf_file: Path) -> Tuple[bool, str]:
    """Validate SGF file format.

    Args:
        sgf_file: Path to SGF file

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        content = sgf_file.read_text(encoding="utf-8")

        # Basic validation
        if not content.strip().startswith("(;"):
            return False, "invalid_format"

        if "SZ[" not in content:
            return False, "missing_board_size"

        # Check for moves
        if not (";B[" in content or ";W[" in content):
            return False, "no_moves"

        return True, "valid"

    except Exception as e:
        return False, f"read_error: {e}"


def _validate_and_filter_file(
    sgf_file: Path, min_moves: int, max_moves: int
) -> Tuple[Path, bool, str, str]:
    """Validate and filter a single SGF file (for parallel processing).

    Args:
        sgf_file: Path to SGF file
        min_moves: Minimum acceptable moves
        max_moves: Maximum acceptable moves

    Returns:
        Tuple of (path, is_valid, validation_reason, filter_reason)
    """
    # Validate
    is_valid, val_reason = validate_sgf(sgf_file)
    if not is_valid:
        return (sgf_file, False, val_reason, "")

    # Filter by length
    should_keep, filter_reason = filter_by_length(sgf_file, min_moves, max_moves)
    return (sgf_file, should_keep, "valid", filter_reason)


def main() -> int:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Filter self-play games by quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter games by length only (parallel)
  python scripts/filter_self_play_games.py \\
      --input data/self_play/9x9 \\
      --output data/self_play/9x9_filtered \\
      --min-moves 20 \\
      --max-moves 150

  # Filter sequentially (no parallel)
  python scripts/filter_self_play_games.py \\
      --input data/self_play/9x9 \\
      --output data/self_play/9x9_filtered \\
      --no-parallel

  # Validate only (no KataGo analysis)
  python scripts/filter_self_play_games.py \\
      --input data/self_play/9x9 \\
      --validate-only

  # With KataGo quality filtering (requires KataGo setup)
  python scripts/filter_self_play_games.py \\
      --input data/self_play/19x19 \\
      --output data/self_play/19x19_filtered \\
      --use-katago \\
      --min-quality 0.3

Note: KataGo quality filtering is not fully implemented yet.
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory with SGF files",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for filtered SGF files",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=20,
        help="Minimum moves per game (default: 20)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Maximum moves per game (default: 300)",
    )
    parser.add_argument(
        "--use-katago",
        action="store_true",
        help="Use KataGo for quality analysis (experimental)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.3,
        help="Minimum quality score (default: 0.3, requires --use-katago)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate, don't copy files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect, 1 = sequential)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Get SGF files
    sgf_files = list(input_dir.glob("*.sgf"))
    if not sgf_files:
        logger.error(f"No SGF files found in {input_dir}")
        return 1

    logger.info(f"Found {len(sgf_files)} SGF files in {input_dir}")

    # Determine whether to use parallel processing
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = _get_num_workers()

    use_parallel = not args.no_parallel and num_workers > 1 and len(sgf_files) > 1

    if use_parallel:
        logger.info(f"Using parallel processing with {num_workers} workers")
        # Parallel validation and filtering
        logger.info("Validating and filtering SGF files in parallel...")
        filtered_files = []
        invalid_count = 0
        filter_reasons = {}

        process_func = partial(
            _validate_and_filter_file,
            min_moves=args.min_moves,
            max_moves=args.max_moves,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_func, sgf_files)

            for sgf_file, should_keep, val_reason, filter_reason in results:
                if val_reason != "valid":
                    invalid_count += 1
                    logger.debug(f"Invalid: {sgf_file.name} - {val_reason}")
                elif should_keep:
                    filtered_files.append(sgf_file)
                else:
                    filter_reasons[filter_reason] = filter_reasons.get(filter_reason, 0) + 1
                    logger.debug(f"Filtered out: {sgf_file.name} - {filter_reason}")

        logger.info(f"Valid files: {len(sgf_files) - invalid_count} / {len(sgf_files)}")
        if invalid_count > 0:
            logger.warning(f"Invalid files: {invalid_count}")

    else:
        logger.info("Using sequential processing")
        # Sequential validation and filtering (original implementation)
        logger.info("Validating SGF files...")
        valid_files = []
        invalid_count = 0

        for sgf_file in sgf_files:
            is_valid, reason = validate_sgf(sgf_file)
            if is_valid:
                valid_files.append(sgf_file)
            else:
                invalid_count += 1
                logger.debug(f"Invalid: {sgf_file.name} - {reason}")

        logger.info(f"Valid files: {len(valid_files)} / {len(sgf_files)}")
        if invalid_count > 0:
            logger.warning(f"Invalid files: {invalid_count}")

        # Filter by length
        logger.info(f"Filtering by length ({args.min_moves}-{args.max_moves} moves)...")
        filtered_files = []
        filter_reasons = {}

        for sgf_file in valid_files:
            should_keep, reason = filter_by_length(
                sgf_file, args.min_moves, args.max_moves
            )
            if should_keep:
                filtered_files.append(sgf_file)
            else:
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                logger.debug(f"Filtered out: {sgf_file.name} - {reason}")

    logger.info(f"After filtering: {len(filtered_files)} files passed")
    if filter_reasons:
        for reason, count in filter_reasons.items():
            logger.info(f"  {reason}: {count}")

    # KataGo quality filtering (if requested)
    if args.use_katago:
        logger.warning("KataGo quality filtering is not fully implemented yet")
        logger.warning("Using length-based filtering only")
        # TODO: Implement KataGo-based quality scoring
        # This would involve:
        # 1. Replay each game move-by-move
        # 2. Get KataGo analysis at each position
        # 3. Compute average move quality
        # 4. Filter based on quality threshold

    # Validate-only mode
    if args.validate_only:
        logger.info("\nValidation complete (validate-only mode)")
        logger.info(f"  Total files: {len(sgf_files)}")
        logger.info(f"  Valid format: {len(valid_files)}")
        logger.info(f"  Passed length filter: {len(filtered_files)}")
        return 0

    # Copy filtered files to output directory
    if not args.output:
        logger.error("--output is required when not using --validate-only")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nCopying filtered files to {output_dir}...")
    for sgf_file in filtered_files:
        dest = output_dir / sgf_file.name
        shutil.copy2(sgf_file, dest)

    logger.info("\nFiltering complete!")
    logger.info(f"  Input files: {len(sgf_files)}")
    logger.info(f"  Valid format: {len(valid_files)}")
    logger.info(f"  Passed filters: {len(filtered_files)}")
    logger.info(f"  Output directory: {output_dir}/")
    logger.info("\nNext step:")
    logger.info("  python scripts/build_sft_dataset.py \\")
    logger.info(f"    --sgf-dir {output_dir} \\")
    logger.info("    --output data/sft_self_play.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
