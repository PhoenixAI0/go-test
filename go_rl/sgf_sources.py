"""SGF file sources and download utilities.

This module provides functions to download SGF files from various public sources.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

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
    "ogs": SGFSource(
        name="Online-Go.com",
        description="OGS public games (requires API key)",
        board_sizes=[9, 13, 19],
        requires_auth=True,
        url="https://online-go.com/",
    ),
    "kgs": SGFSource(
        name="KGS Archives",
        description="KGS game archives (downloadable via HTTP)",
        board_sizes=[9, 13, 19],
        requires_auth=False,
        url="https://www.gokgs.com/gameArchives.html",
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


def download_kgs_games(
    output_dir: Path,
    board_size: int = 19,
    max_games: int = 100,
    year: Optional[int] = None,
) -> int:
    """Download games from KGS archives.

    Note: This is a placeholder implementation. KGS archives are organized
    by year and player, so you would need to:
    1. Parse the archives index page
    2. Select specific years/players
    3. Download individual game files

    Args:
        output_dir: Directory to save SGF files
        board_size: Board size to filter (9, 13, or 19)
        max_games: Maximum number of games to download
        year: Year to download from (None = most recent)

    Returns:
        Number of games downloaded
    """
    logger.warning(
        "KGS download not fully implemented. "
        "KGS archives require manual selection of years/players. "
        "Visit https://www.gokgs.com/gameArchives.html to browse archives."
    )
    return 0


def download_ogs_games(
    output_dir: Path,
    board_size: int = 19,
    max_games: int = 100,
    api_key: Optional[str] = None,
    min_rank: Optional[str] = None,
) -> int:
    """Download games from Online-Go.com.

    Note: This requires an OGS API key. To get one:
    1. Create an account at https://online-go.com/
    2. Generate an API key in your settings
    3. Pass it via --api-key or OGS_API_KEY environment variable

    Args:
        output_dir: Directory to save SGF files
        board_size: Board size to filter (9, 13, or 19)
        max_games: Maximum number of games to download
        api_key: OGS API key for authentication
        min_rank: Minimum rank filter (e.g., "5k", "1d")

    Returns:
        Number of games downloaded
    """
    if not api_key:
        logger.error(
            "OGS download requires API key. "
            "Get one from https://online-go.com/ and pass via --api-key"
        )
        return 0

    logger.warning(
        "OGS download not fully implemented. "
        "You can download your own games manually from https://online-go.com/"
    )
    return 0


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

    except Exception as e:
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

    except Exception as e:
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
    print("  3. Prepare data: ./prepare_data.sh\n")
