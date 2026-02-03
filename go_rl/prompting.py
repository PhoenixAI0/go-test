from __future__ import annotations

import re

from typing import List, Optional

from .board import BoardState


def parse_prompt(prompt: str) -> BoardState:
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("board_size:"):
        raise ValueError("Prompt missing board_size header")
    size = int(lines[0].split(":", 1)[1].strip())
    board = BoardState.empty(size)
    for line in lines[1:]:
        if line == "move:":
            break
        player, move = line.split(":", 1)
        board.apply_move(player.strip().upper(), move.strip())
    return board


_MOVE_PATTERN = re.compile(r"\b([A-HJ-T])\s*([1-9]|1[0-9])\b", re.IGNORECASE)
_PASS_PATTERN = re.compile(r"\bPASS\b", re.IGNORECASE)


def build_prompt(board: BoardState, player: Optional[str] = None) -> str:
    if player is not None:
        normalized = player.strip().upper()
        if normalized not in {"B", "W"}:
            raise ValueError(f"Unknown player: {player}")
    return board.to_prompt()


def parse_move(response: str) -> str:
    """Parse a GTP move or PASS from a model response."""
    if not response:
        raise ValueError("Empty response")
    if _PASS_PATTERN.search(response):
        return "PASS"
    match = _MOVE_PATTERN.search(response)
    if match:
        return f"{match.group(1).upper()}{match.group(2)}"
    raise ValueError(f"Could not parse move from response: {response!r}")
