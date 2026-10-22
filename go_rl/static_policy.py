"""A fitted coordinate-frequency control for board-dependent model policies."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .board import BoardState, gtp_to_coords
from .data_prep import _transform_coords, board_from_example
from .model_interface import action_to_move, legal_action_mask
from .rules import legality_superko


def fit_response_frequencies(
    records: Iterable[Mapping[str, Any]], *, board_size: int, symmetries: int = 1,
) -> tuple[list[int], int]:
    """Count explicit teacher responses; do not use dense policies or eval labels.

    Every input row contributes equally, matching empirical dataset frequency.
    Optional D4 expansion rotates/reflects coordinates without changing color.
    """
    if symmetries not in (1, 8):
        raise ValueError("Frequency symmetries must be 1 or 8")
    counts = [0] * (board_size * board_size + 1)
    rows = 0
    for record in records:
        if record.get("split") != "train":
            raise ValueError("Frequency fitting requires explicitly pinned training records")
        board = board_from_example(record)
        if board.size != board_size:
            raise ValueError("Frequency source and evaluation board sizes must match")
        response = record.get("response")
        if not isinstance(response, str) or not response:
            raise ValueError("Frequency source needs an explicit teacher response")
        if response.upper() not in board.legal_moves(superko=legality_superko(record["rules"])):
            raise ValueError("Frequency source response is illegal in its recorded state")
        coords = gtp_to_coords(response, board_size)
        for symmetry in range(symmetries):
            transformed = _transform_coords(coords, board_size, symmetry)
            action = board_size * board_size if transformed is None else transformed[0] * board_size + transformed[1]
            counts[action] += 1
        rows += 1
    if not rows:
        raise ValueError("Frequency source is empty")
    return counts, rows


def frequency_move(counts: Sequence[int], board: BoardState, *, rules: Any) -> str:
    """Highest training count among legal actions, with row-major/pass-last ties."""
    if (len(counts) != board.size * board.size + 1 or
            any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts)
            or not sum(counts)):
        raise ValueError("Frequency counts must be nonnegative integer action counts with positive mass")
    legal = legal_action_mask(board, superko=legality_superko(rules))
    action = max((index for index, allowed in enumerate(legal) if allowed), key=lambda index: (counts[index], -index))
    return action_to_move(action, board.size)
