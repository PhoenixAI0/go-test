from __future__ import annotations

from typing import List

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


def build_prompt(board: BoardState) -> str:
    return board.to_prompt()
