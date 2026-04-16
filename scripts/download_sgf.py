#!/usr/bin/env python3
"""Inspect and validate SGF files for training Go models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.sgf_sources import list_sources, print_manual_instructions, validate_sgf_directory
from venv_utils import ensure_virtualenv


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s: %(message)s",
    )


def main() -> int:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Inspect and validate SGF files for Go RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate existing SGF files
  uv run python scripts/download_sgf.py --validate-only --board-size 9

  # List available sources
  uv run python scripts/download_sgf.py --list-sources

  # Show manual download instructions
  uv run python scripts/download_sgf.py --manual-instructions --board-size 9

Note: SGF sources are listed for manual download or export. This script
      validates local files and points you to the available sources.
        """,
    )

    parser.add_argument(
        "--board-size",
        type=int,
        default=9,
        choices=[9, 13, 19],
        help="Board size (9, 13, or 19, default: 9)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default: data/sgf_<board_size>)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing SGF files",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available SGF sources",
    )
    parser.add_argument(
        "--manual-instructions",
        action="store_true",
        help="Print manual download instructions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/sgf_{args.board_size}x{args.board_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle list sources
    if args.list_sources:
        print(f"\nAvailable SGF sources for {args.board_size}x{args.board_size}:\n")
        sources = list_sources(args.board_size)
        for source in sources:
            print(f"  {source.name}")
            print(f"    {source.description}")
            print(f"    URL: {source.url}")
            print(f"    Requires auth: {source.requires_auth}")
            print()
        return 0

    # Handle manual instructions
    if args.manual_instructions:
        print_manual_instructions(args.board_size, output_dir)
        return 0

    # Handle validate only
    if args.validate_only:
        logger.info(f"Validating SGF files in {output_dir}")
        valid, invalid = validate_sgf_directory(
            output_dir,
            expected_board_size=args.board_size,
            verbose=args.verbose,
        )

        print(f"\nValidation Results:")
        print(f"  Valid:   {valid}")
        print(f"  Invalid: {invalid}")
        print(f"  Total:   {valid + invalid}")

        if valid == 0:
            print(f"\n⚠️  No valid SGF files found in {output_dir}")
            print("\nTo download SGF files:")
            print("  1. Use --manual-instructions to see download sources")
            print(f"  2. Copy SGF files to {output_dir}/")
            print("  3. Run with --validate-only to verify")
            return 1

        print(f"\n✓ Found {valid} valid SGF files")
        print("\nNext step: Run ./scripts/prepare_data.sh to create training data")
        return 0

    # Default: show help and manual instructions
    print("\nNo action specified. Use --help for usage information.\n")
    print_manual_instructions(args.board_size, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
