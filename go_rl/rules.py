"""Go rules and legality checking with different rule variants.

This module provides configurable rule checking for Go, including:
- Basic legality (position occupancy only)
- Ko rule (simple ko - immediate recapture forbidden)
- Superko rule (positional superko - any previous position forbidden)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple


class LegalityLevel(Enum):
    """Legality checking levels."""

    BASIC = "basic"  # Only check if position is empty
    KO = "ko"  # Check for simple ko (immediate recapture)
    SUPERKO = "superko"  # Check for positional superko (any repeat)


def board_to_hash(board: List[List[int]]) -> int:
    """Convert board state to hash for comparison.

    Args:
        board: 2D board array

    Returns:
        Hash of board state
    """
    # Simple hash using tuple representation
    return hash(tuple(tuple(row) for row in board))


def boards_equal(board1: List[List[int]], board2: List[List[int]]) -> bool:
    """Check if two boards are equal.

    Args:
        board1: First board
        board2: Second board

    Returns:
        True if boards are identical
    """
    if len(board1) != len(board2):
        return False

    for row1, row2 in zip(board1, board2):
        if len(row1) != len(row2):
            return False
        for cell1, cell2 in zip(row1, row2):
            if cell1 != cell2:
                return False

    return True


class BoardHistory:
    """Track board positions for superko detection."""

    def __init__(self, max_size: int = 1000):
        """Initialize board history.

        Args:
            max_size: Maximum number of positions to track
        """
        self.positions: List[int] = []
        self.max_size = max_size

    def add(self, board: List[List[int]]) -> None:
        """Add a board position to history.

        Args:
            board: Board state to add
        """
        board_hash = board_to_hash(board)
        self.positions.append(board_hash)

        # Limit size to prevent unbounded growth
        if len(self.positions) > self.max_size:
            self.positions = self.positions[-self.max_size :]

    def contains(self, board: List[List[int]]) -> bool:
        """Check if a board position exists in history.

        Args:
            board: Board state to check

        Returns:
            True if position was seen before
        """
        board_hash = board_to_hash(board)
        return board_hash in self.positions

    def clear(self) -> None:
        """Clear history."""
        self.positions = []

    def __len__(self) -> int:
        """Get number of positions in history."""
        return len(self.positions)


def is_superko_violation(
    new_board: List[List[int]], board_history: BoardHistory
) -> bool:
    """Check if a new board state violates superko rule.

    Args:
        new_board: Proposed new board state
        board_history: History of previous board states

    Returns:
        True if this position was seen before (superko violation)
    """
    return board_history.contains(new_board)


def is_simple_ko_violation(
    new_board: List[List[int]], previous_board: Optional[List[List[int]]]
) -> bool:
    """Check if a new board state violates simple ko rule.

    Args:
        new_board: Proposed new board state
        previous_board: Immediately previous board state

    Returns:
        True if new board matches previous board (simple ko violation)
    """
    if previous_board is None:
        return False

    return boards_equal(new_board, previous_board)


def validate_legality(
    new_board: List[List[int]],
    previous_board: Optional[List[List[int]]],
    board_history: Optional[BoardHistory],
    legality_level: LegalityLevel,
) -> Tuple[bool, Optional[str]]:
    """Validate move legality according to specified rules.

    Args:
        new_board: Proposed new board state
        previous_board: Immediately previous board state
        board_history: History of all board states
        legality_level: Level of legality checking

    Returns:
        Tuple of (is_legal, violation_reason)
    """
    # Basic legality doesn't check ko
    if legality_level == LegalityLevel.BASIC:
        return True, None

    # Simple ko check
    if legality_level == LegalityLevel.KO:
        if is_simple_ko_violation(new_board, previous_board):
            return False, "simple_ko_violation"
        return True, None

    # Superko check
    if legality_level == LegalityLevel.SUPERKO:
        # First check simple ko (faster)
        if is_simple_ko_violation(new_board, previous_board):
            return False, "simple_ko_violation"

        # Then check superko
        if board_history and is_superko_violation(new_board, board_history):
            return False, "superko_violation"

        return True, None

    # Unknown legality level
    return True, None


def format_legality_error(reason: Optional[str]) -> str:
    """Format a legality violation reason as a human-readable message.

    Args:
        reason: Violation reason code

    Returns:
        Human-readable error message
    """
    messages = {
        "simple_ko_violation": "Illegal move: simple ko (cannot immediately recapture)",
        "superko_violation": "Illegal move: positional superko (position was seen before)",
    }

    return messages.get(reason, f"Illegal move: {reason}")
