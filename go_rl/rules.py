from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple


class LegalityLevel(Enum):
    BASIC = "basic"
    KO = "ko"
    SUPERKO = "superko"


def board_to_hash(board: List[List[int]]) -> int:
    return hash(tuple(tuple(row) for row in board))


def boards_equal(board1: List[List[int]], board2: List[List[int]]) -> bool:
    return board1 == board2


class BoardHistory:
    def __init__(self, max_size: int = 1000):
        self.positions: List[int] = []
        self.max_size = max_size

    def add(self, board: List[List[int]]) -> None:
        self.positions.append(board_to_hash(board))
        if len(self.positions) > self.max_size:
            del self.positions[: len(self.positions) - self.max_size]

    def contains(self, board: List[List[int]]) -> bool:
        return board_to_hash(board) in self.positions

    def clear(self) -> None:
        self.positions.clear()

    def __len__(self) -> int:
        return len(self.positions)


def is_superko_violation(
    new_board: List[List[int]], board_history: BoardHistory
) -> bool:
    return board_history.contains(new_board)


def is_simple_ko_violation(
    new_board: List[List[int]], previous_board: Optional[List[List[int]]]
) -> bool:
    return previous_board is not None and boards_equal(new_board, previous_board)


def validate_legality(
    new_board: List[List[int]],
    previous_board: Optional[List[List[int]]],
    board_history: Optional[BoardHistory],
    legality_level: LegalityLevel,
) -> Tuple[bool, Optional[str]]:
    if legality_level == LegalityLevel.BASIC:
        return True, None

    if is_simple_ko_violation(new_board, previous_board):
        return False, "simple_ko_violation"

    if legality_level == LegalityLevel.SUPERKO and board_history:
        if is_superko_violation(new_board, board_history):
            return False, "superko_violation"
    return True, None


def format_legality_error(reason: Optional[str]) -> str:
    messages = {
        "simple_ko_violation": "Illegal move: simple ko (cannot immediately recapture)",
        "superko_violation": "Illegal move: positional superko (position was seen before)",
    }

    return messages.get(reason, f"Illegal move: {reason}")
