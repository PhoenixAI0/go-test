"""SGF source metadata and validation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SGFSource:
    """Definition of an SGF file source."""

    name: str
    description: str
    board_sizes: List[int]
    requires_auth: bool
    url: str


# Public SGF sources
SOURCES = {
    "kgs": SGFSource(
        name="KGS Archives",
        description="KGS archives for manual download",
        board_sizes=[9, 13, 19],
        requires_auth=False,
        url="https://www.gokgs.com/gameArchives.html",
    ),
    "ogs": SGFSource(
        name="Online-Go.com",
        description="OGS public games (manual export or API key required)",
        board_sizes=[9, 13, 19],
        requires_auth=True,
        url="https://online-go.com/",
    ),
    "waltheri": SGFSource(
        name="Waltheri Pattern Search",
        description="Historical professional games (web scraping required)",
        board_sizes=[19],
        requires_auth=False,
        url="https://ps.waltheri.net/",
    ),
    "gogod": SGFSource(
        name="GoGoD",
        description="Commercial database (manual download required)",
        board_sizes=[19],
        requires_auth=True,
        url="https://gogodonline.co.uk/",
    ),
}


def validate_sgf_file(file_path: Path) -> bool:
    """Validate that a file is a valid SGF.

    Args:
        file_path: Path to SGF file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic validation: must start with (;
        if not content.strip().startswith("(;"):
            logger.debug(f"Invalid SGF format: {file_path}")
            return False

        # Must contain board size
        if "SZ[" not in content:
            logger.debug(f"Missing board size: {file_path}")
            return False

        return True

    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Error validating {file_path}: {e}")
        return False


def extract_board_size(file_path: Path) -> Optional[int]:
    """Extract board size from an SGF file.

    Args:
        file_path: Path to SGF file

    Returns:
        Board size (9, 13, 19, etc.) or None if not found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for SZ[X] tag
        import re

        match = re.search(r"SZ\[(\d+)\]", content)
        if match:
            return int(match.group(1))

        return None

    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Error extracting board size from {file_path}: {e}")
        return None


def validate_sgf_directory(
    directory: Path,
    expected_board_size: Optional[int] = None,
    verbose: bool = False,
) -> tuple[int, int]:
    """Validate all SGF files in a directory.

    Args:
        directory: Directory containing SGF files
        expected_board_size: Expected board size (None = any size)
        verbose: Print detailed validation info

    Returns:
        Tuple of (valid_count, invalid_count)
    """
    sgf_files = list(directory.glob("*.sgf"))
    valid_count = 0
    invalid_count = 0

    logger.info(f"Validating {len(sgf_files)} SGF files in {directory}")

    for sgf_file in sgf_files:
        if not validate_sgf_file(sgf_file):
            invalid_count += 1
            if verbose:
                logger.warning(f"Invalid SGF: {sgf_file.name}")
            continue

        # Check board size if specified
        if expected_board_size is not None:
            board_size = extract_board_size(sgf_file)
            if board_size != expected_board_size:
                invalid_count += 1
                if verbose:
                    logger.warning(
                        f"Wrong board size ({board_size} != {expected_board_size}): "
                        f"{sgf_file.name}"
                    )
                continue

        valid_count += 1

    logger.info(f"Validation: {valid_count} valid, {invalid_count} invalid")
    return valid_count, invalid_count


def list_sources(board_size: Optional[int] = None) -> List[SGFSource]:
    """List available SGF sources.

    Args:
        board_size: Filter by board size (None = all)

    Returns:
        List of available sources
    """
    sources = list(SOURCES.values())

    if board_size is not None:
        sources = [s for s in sources if board_size in s.board_sizes]

    return sources


def print_manual_instructions(board_size: int, output_dir: Path) -> None:
    """Print manual download instructions for SGF files.

    Args:
        board_size: Target board size
        output_dir: Where to save downloaded files
    """
    print("\n" + "=" * 60)
    print(f"Manual Download Instructions for {board_size}x{board_size} SGF Files")
    print("=" * 60 + "\n")

    sources = list_sources(board_size)

    print(f"Save SGF files to: {output_dir}/\n")

    for i, source in enumerate(sources, 1):
        print(f"{i}. {source.name}")
        print(f"   {source.description}")
        print(f"   URL: {source.url}")
        if source.requires_auth:
            print("   ⚠️  Requires authentication or manual download")
        print()

    print("After downloading SGF files:")
    print(f"  1. Copy them to {output_dir}/")
    print("  2. Validate: python scripts/download_sgf.py --validate-only")
    print("  3. Prepare data: ./scripts/prepare_data.sh\n")
