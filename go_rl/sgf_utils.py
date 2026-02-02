from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple

from .board import BoardState, coords_to_gtp

logger = logging.getLogger(__name__)


def _get_num_workers() -> int:
    """Get optimal number of worker processes for parallel processing."""
    try:
        cpu_count = mp.cpu_count()
        return max(1, cpu_count - 1)
    except NotImplementedError:
        return 4


@dataclass
class SgfGame:
    board_size: int
    moves: List[Tuple[str, Tuple[int, int] | None]]
    setup_stones: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)


def _require_sgfmill() -> None:
    try:
        import sgfmill  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sgfmill is required to parse SGF files.") from exc


def load_sgf(path: str) -> SgfGame:
    _require_sgfmill()
    from sgfmill import sgf

    logger.debug("Loading SGF file %s", path)
    with open(path, "rb") as handle:
        collection = sgf.Sgf_collection.from_bytes(handle.read())
    if not collection:
        raise ValueError("SGF file contains no games")
    game = collection[0]
    root = game.get_root()
    board_size = int(root.get("SZ", 19))
    moves: List[Tuple[str, Tuple[int, int] | None]] = []
    setup_stones: List[Tuple[str, Tuple[int, int]]] = []

    setup_black = root.get("AB") or []
    setup_white = root.get("AW") or []
    for coords in setup_black:
        setup_stones.append(("B", coords))
    for coords in setup_white:
        setup_stones.append(("W", coords))

    for node in game.get_main_sequence()[1:]:
        if node.has_move("b"):
            color = "B"
            coords = node.get_move("b")
        elif node.has_move("w"):
            color = "W"
            coords = node.get_move("w")
        else:
            continue
        moves.append((color, coords))
    logger.debug("Parsed SGF file %s with %d moves", path, len(moves))
    return SgfGame(
        board_size=board_size,
        moves=moves,
        setup_stones=setup_stones,
    )


def sample_positions(game: SgfGame, max_positions: int) -> List[dict]:
    board = BoardState.empty(game.board_size)
    samples: List[dict] = []
    logger.debug(
        "Sampling up to %d positions for %dx%d game",
        max_positions,
        game.board_size,
        game.board_size,
    )
    for player, coords in game.setup_stones:
        gtp_move = coords_to_gtp(coords, game.board_size)
        board.apply_move(player, gtp_move)
    for player, coords in game.moves:
        if coords is None:
            gtp_move = "PASS"
        else:
            gtp_move = coords_to_gtp(coords, game.board_size)
        samples.append(
            {
                "prompt": board.to_prompt(),
                "target": gtp_move,
                "player": player,
                "board_size": game.board_size,
            }
        )
        board.apply_move(player, gtp_move)
        if len(samples) >= max_positions:
            break
    logger.debug("Collected %d samples from game", len(samples))
    return samples


def _process_sgf_file(path: Path, max_positions_per_file: int) -> List[dict]:
    """Process a single SGF file and return samples.

    This function is designed to be pickled for multiprocessing.

    Args:
        path: Path to SGF file
        max_positions_per_file: Maximum positions to sample from this file

    Returns:
        List of samples from the file
    """
    try:
        game = load_sgf(str(path))
        samples = sample_positions(game, max_positions=max_positions_per_file)
        return samples
    except Exception as exc:
        logger.warning("Skipping SGF file %s: %s", path, exc)
        return []


def iter_sgf_samples(
    sgf_dir: str, max_positions: int, num_workers: int = 0, use_parallel: bool = True
) -> Iterable[dict]:
    """Iterate over SGF samples from a directory.

    Args:
        sgf_dir: Directory containing SGF files
        max_positions: Maximum total positions to collect
        num_workers: Number of parallel workers (0 = auto-detect, 1 = sequential)
        use_parallel: Whether to use parallel processing

    Yields:
        Sample dictionaries with prompt, target, player, board_size
    """
    logger.info("Scanning SGF directory %s", sgf_dir)
    sgf_files = sorted(Path(sgf_dir).glob("**/*.sgf"))
    total_files = len(sgf_files)
    logger.info("Found %d SGF files", total_files)

    if total_files == 0:
        logger.warning("No SGF files found in %s", sgf_dir)
        return

    # Determine number of workers
    if num_workers == 0:
        num_workers = _get_num_workers()

    # Use parallel processing for multiple files
    if use_parallel and num_workers > 1 and total_files > 1:
        logger.info("Using parallel processing with %d workers", num_workers)
        yield from _iter_sgf_samples_parallel(sgf_files, max_positions, num_workers)
    else:
        logger.info("Using sequential processing")
        yield from _iter_sgf_samples_sequential(sgf_files, max_positions)


def _iter_sgf_samples_sequential(
    sgf_files: List[Path], max_positions: int
) -> Iterable[dict]:
    """Sequential SGF processing (original implementation).

    Args:
        sgf_files: List of SGF file paths
        max_positions: Maximum total positions to collect

    Yields:
        Sample dictionaries
    """
    collected = 0
    for path in sgf_files:
        if collected >= max_positions:
            break
        try:
            game = load_sgf(str(path))
        except Exception as exc:
            logger.warning("Skipping SGF file %s: %s", path, exc)
            continue
        remaining = max_positions - collected
        logger.debug("Sampling %d remaining positions from %s", remaining, path)
        try:
            samples = sample_positions(game, max_positions=remaining)
        except Exception as exc:
            logger.warning("Skipping SGF file %s: %s", path, exc)
            continue
        for sample in samples:
            yield sample
            collected += 1
            if collected >= max_positions:
                break


def _iter_sgf_samples_parallel(
    sgf_files: List[Path], max_positions: int, num_workers: int
) -> Iterable[dict]:
    """Parallel SGF processing using multiprocessing.

    Args:
        sgf_files: List of SGF file paths
        max_positions: Maximum total positions to collect
        num_workers: Number of parallel workers

    Yields:
        Sample dictionaries
    """
    # Estimate positions per file for better work distribution
    total_files = len(sgf_files)
    estimated_positions_per_file = max(1, max_positions // total_files)

    # Process files in batches to avoid memory issues
    batch_size = max(1, num_workers * 4)
    collected = 0

    with mp.Pool(processes=num_workers) as pool:
        # Process in batches
        for batch_start in range(0, total_files, batch_size):
            if collected >= max_positions:
                break

            batch_end = min(batch_start + batch_size, total_files)
            batch_files = sgf_files[batch_start:batch_end]

            remaining = max_positions - collected
            positions_per_file = max(1, remaining // len(batch_files))

            logger.debug(
                "Processing batch %d-%d of %d files (up to %d positions per file)",
                batch_start,
                batch_end,
                total_files,
                positions_per_file,
            )

            # Process batch in parallel
            process_func = partial(_process_sgf_file, max_positions_per_file=positions_per_file)
            results = pool.map(process_func, batch_files)

            # Yield samples from results
            for samples in results:
                if collected >= max_positions:
                    break
                for sample in samples:
                    if collected >= max_positions:
                        break
                    yield sample
                    collected += 1

            logger.debug("Collected %d / %d positions so far", collected, max_positions)


def load_sgf_dir(sgf_dir: str, max_positions: int) -> List[dict]:
    samples = list(iter_sgf_samples(sgf_dir, max_positions=max_positions))
    logger.info("Loaded %d samples from %s", len(samples), sgf_dir)
    return samples
