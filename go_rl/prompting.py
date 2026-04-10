from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

from .board import BoardState, _EMPTY, _STONE


def parse_prompt(prompt: str) -> BoardState:
    lines = [line.strip() for line in prompt.strip().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("board_size:"):
        raise ValueError("Prompt missing board_size header")
    size = int(lines[0].split(":", 1)[1].strip())
    if any(line.startswith("board:") for line in lines[1:]):
        return _parse_matrix_prompt(lines, size)
    return _parse_move_history_prompt(lines, size)


def _parse_move_history_prompt(lines: List[str], size: int) -> BoardState:
    board = BoardState.empty(size)
    for line in lines[1:]:
        if line == "move:":
            break
        player, move = line.split(":", 1)
        board.apply_move(player.strip().upper(), move.strip())
    return board


def _parse_matrix_prompt(lines: List[str], size: int) -> BoardState:
    to_play: Optional[str] = None
    board_start: Optional[int] = None
    for index, line in enumerate(lines[1:], start=1):
        if line.startswith("to_play:"):
            to_play = line.split(":", 1)[1].strip().upper()
        elif line == "board:":
            board_start = index + 1
            break
    if to_play not in {"B", "W"}:
        raise ValueError("Prompt missing to_play header")
    if board_start is None:
        raise ValueError("Prompt missing board block")

    matrix_rows: List[List[int]] = []
    for line in lines[board_start:]:
        if line == "move:":
            break
        row_text = line.replace(" ", "").upper()
        if not row_text:
            continue
        if row_text.isalpha() and len(row_text) == size:
            continue
        if ":" in row_text:
            continue
        if row_text and row_text[0].isdigit():
            parts = row_text.split(".", 1)
            if len(parts) == 2:
                row_text = "." + parts[1]
            else:
                continue
        if len(row_text) != size:
            raise ValueError("Prompt row width does not match board size")
        row: List[int] = []
        for char in row_text:
            if char == ".":
                row.append(_EMPTY)
            elif char in {"X", "B"}:
                row.append(_STONE["B"])
            elif char in {"O", "W"}:
                row.append(_STONE["W"])
            else:
                raise ValueError(f"Unknown board cell: {char}")
        matrix_rows.append(row)
    if len(matrix_rows) != size:
        raise ValueError("Prompt board height does not match board size")
    return BoardState.from_matrix(size=size, board=matrix_rows, next_player=to_play)


_MOVE_PATTERN = re.compile(r"\b([A-HJ-T])\s*([1-9]|1[0-9])\b", re.IGNORECASE)
_PASS_PATTERN = re.compile(r"\bPASS\b", re.IGNORECASE)


def build_prompt(board: BoardState, player: Optional[str] = None) -> str:
    if player is not None:
        normalized = player.strip().upper()
        if normalized not in {"B", "W"}:
            raise ValueError(f"Unknown player: {player}")
    return board.to_prompt(player=player)


def build_generation_inputs(
    tokenizer: object,
    prompt: str,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chat_kwargs = chat_template_kwargs or {}
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **chat_kwargs,
            )
            if isinstance(rendered, dict):
                return rendered
        except TypeError:
            pass
    return tokenizer(prompt, return_tensors="pt")


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
