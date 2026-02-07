from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

_DEPRECATION_MESSAGE = (
    "Python SGF parsing is deprecated and no longer supported. "
    "Use the Go parser via build_sft_dataset/run_go_parser and scripts/*dataset*.py."
)


@dataclass
class SgfGame:
    board_size: int
    moves: List[Tuple[str, Tuple[int, int] | None]]
    setup_stones: List[Tuple[str, Tuple[int, int]]] = field(default_factory=list)


def _deprecated() -> None:
    raise RuntimeError(_DEPRECATION_MESSAGE)


def load_sgf(path: str) -> SgfGame:
    _ = path
    _deprecated()


def _require_sgfmill() -> None:
    _deprecated()


def sample_positions(game: SgfGame, max_positions: int) -> List[dict]:
    _ = game
    _ = max_positions
    _deprecated()


def iter_sgf_samples(
    sgf_dir: str, max_positions: int, num_workers: int = 0, use_parallel: bool = True
) -> Iterable[dict]:
    _ = sgf_dir
    _ = max_positions
    _ = num_workers
    _ = use_parallel
    _deprecated()
    return iter(())


def load_sgf_dir(sgf_dir: str, max_positions: int) -> List[dict]:
    _ = sgf_dir
    _ = max_positions
    _deprecated()
