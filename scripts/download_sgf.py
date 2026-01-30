#!/usr/bin/env python3
"""Download SGF files from public sources.

This script provides automated and semi-automated ways to download SGF files
for training Go models.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.sgf_sources import (
    SOURCES,
    download_kgs_games,
    download_ogs_games,
    list_sources,
    print_manual_instructions,
    validate_sgf_directory,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SGF files for Go RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate existing SGF files
  python scripts/download_sgf.py --validate-only --board-size 9

  # List available sources
  python scripts/download_sgf.py --list-sources

  # Download from KGS (placeholder)
  python scripts/download_sgf.py --source kgs --board-size 19 --count 100

  # Download from OGS (requires API key)
  python scripts/download_sgf.py --source ogs --api-key YOUR_KEY --count 100

  # Show manual download instructions
  python scripts/download_sgf.py --manual-instructions --board-size 9

Note: Most sources require manual download or API keys. This script provides
      validation and guidance for obtaining SGF files.
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
        "--count",
        type=int,
        default=100,
        help="Number of games to download (default: 100)",
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()),
        help="Source to download from (kgs, ogs, waltheri)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authenticated sources (OGS)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing SGF files, don't download",
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
        print("\nNext step: Run ./prepare_data.sh to create training data")
        return 0

    # Handle download
    if args.source:
        logger.info(
            f"Downloading {args.count} games from {args.source} "
            f"(board size: {args.board_size})"
        )

        downloaded = 0
        if args.source == "kgs":
            downloaded = download_kgs_games(
                output_dir=output_dir,
                board_size=args.board_size,
                max_games=args.count,
            )
        elif args.source == "ogs":
            api_key = args.api_key or os.environ.get("OGS_API_KEY")
            if not api_key:
                logger.error(
                    "OGS requires API key. Pass via --api-key or OGS_API_KEY env var"
                )
                return 1
            downloaded = download_ogs_games(
                output_dir=output_dir,
                board_size=args.board_size,
                max_games=args.count,
                api_key=api_key,
            )
        elif args.source == "waltheri":
            logger.warning(
                "Waltheri download not implemented. "
                "Visit https://ps.waltheri.net/ to download games manually."
            )
            downloaded = 0

        if downloaded > 0:
            logger.info(f"Successfully downloaded {downloaded} games")

            # Validate downloaded files
            logger.info("Validating downloaded files...")
            valid, invalid = validate_sgf_directory(
                output_dir,
                expected_board_size=args.board_size,
                verbose=args.verbose,
            )

            if valid > 0:
                print(f"\n✓ Downloaded and validated {valid} SGF files")
                print("\nNext step: Run ./prepare_data.sh to create training data")
                return 0
            else:
                logger.error("No valid SGF files after download")
                return 1
        else:
            logger.warning("No games downloaded (source may require manual download)")
            print_manual_instructions(args.board_size, output_dir)
            return 1

    # Default: show help and manual instructions
    print("\nNo action specified. Use --help for usage information.\n")
    print_manual_instructions(args.board_size, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
