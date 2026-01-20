from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from .board import BoardState, coords_to_gtp

logger = logging.getLogger(__name__)


@dataclass
class SgfGame:
    board_size: int
    moves: List[Tuple[str, Tuple[int, int] | None]]


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
    game = collection[0]
    root = game.get_root()
    board_size = int(root.get("SZ", 19))
    moves: List[Tuple[str, Tuple[int, int] | None]] = []

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
    return SgfGame(board_size=board_size, moves=moves)


def sample_positions(game: SgfGame, max_positions: int) -> List[dict]:
    board = BoardState.empty(game.board_size)
    samples: List[dict] = []
    logger.debug(
        "Sampling up to %d positions for %dx%d game",
        max_positions,
        game.board_size,
        game.board_size,
    )
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


def load_sgf_dir(sgf_dir: str, max_positions: int) -> List[dict]:
    samples: List[dict] = []
    logger.info("Scanning SGF directory %s", sgf_dir)
    for path in Path(sgf_dir).glob("**/*.sgf"):
        game = load_sgf(str(path))
        remaining = max_positions - len(samples)
        if remaining <= 0:
            break
        logger.debug(
            "Sampling %d remaining positions from %s",
            remaining,
            path,
        )
        samples.extend(sample_positions(game, max_positions=remaining))
    logger.info("Loaded %d samples from %s", len(samples), sgf_dir)
    return samples
