from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

from .board import BoardState, coords_to_gtp

logger = logging.getLogger(__name__)


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


def iter_sgf_samples(sgf_dir: str, max_positions: int) -> Iterable[dict]:
    logger.info("Scanning SGF directory %s", sgf_dir)
    collected = 0
    for path in sorted(Path(sgf_dir).glob("**/*.sgf")):
        if collected >= max_positions:
            break
        try:
            game = load_sgf(str(path))
        except Exception as exc:
            logger.warning("Skipping SGF file %s: %s", path, exc)
            continue
        remaining = max_positions - collected
        logger.debug(
            "Sampling %d remaining positions from %s",
            remaining,
            path,
        )
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


def load_sgf_dir(sgf_dir: str, max_positions: int) -> List[dict]:
    samples = list(iter_sgf_samples(sgf_dir, max_positions=max_positions))
    logger.info("Loaded %d samples from %s", len(samples), sgf_dir)
    return samples
