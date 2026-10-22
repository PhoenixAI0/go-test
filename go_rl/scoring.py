"""Deterministic current-board area counting, without dead-stone inference.

This implements stones plus empty regions bordered by one color. A terminal
result requires two passes, area scoring, no tax/button/handicap bonus, and an
explicit capture-before-passing convention. It is not Japanese scoring or a
replacement for agreed dead-stone removal. KataGo's pass-alive simplifications
must be audited separately before treating its result as identical.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from .board import BoardState
from .rules import normalize_rules


@dataclass(frozen=True)
class AreaCount:
    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    neutral: int

    @property
    def black(self) -> int:
        return self.black_stones + self.black_territory

    @property
    def white(self) -> int:
        return self.white_stones + self.white_territory


def count_board_area(matrix: Sequence[Sequence[int]]) -> AreaCount:
    """Flood-fill empty regions; all present stones remain on the board."""
    size = len(matrix)
    if size < 1 or any(len(row) != size for row in matrix):
        raise ValueError("Area counting requires a nonempty square board")
    if any(type(value) is not int or value not in (0, 1, 2) for row in matrix for value in row):
        raise ValueError("Board points must be integers 0, 1, or 2")
    stones = [0, 0, 0]
    territory = [0, 0, 0]
    visited = set()
    for row in range(size):
        for column in range(size):
            value = matrix[row][column]
            if value:
                stones[value] += 1
                continue
            if (row, column) in visited:
                continue
            frontier = [(row, column)]
            visited.add((row, column))
            region_size = 0
            border = set()
            while frontier:
                current_row, current_column = frontier.pop()
                region_size += 1
                for next_row, next_column in ((current_row - 1, current_column),
                                              (current_row + 1, current_column),
                                              (current_row, current_column - 1),
                                              (current_row, current_column + 1)):
                    if not (0 <= next_row < size and 0 <= next_column < size):
                        continue
                    neighbor = matrix[next_row][next_column]
                    if neighbor:
                        border.add(neighbor)
                    elif (next_row, next_column) not in visited:
                        visited.add((next_row, next_column))
                        frontier.append((next_row, next_column))
            owner = next(iter(border)) if len(border) == 1 else 0
            territory[owner] += region_size
    return AreaCount(stones[1], stones[2], territory[1], territory[2], territory[0])


def validate_board_area_rules(rules: str | Mapping[str, object]) -> None:
    normalized = normalize_rules(rules)
    if (normalized["scoring"] != "AREA" or normalized["tax"] != "NONE"
            or normalized["hasButton"] or normalized["whiteHandicapBonus"] != "0"
            or normalized["friendlyPassOk"]):
        raise ValueError("Exact board-area results require AREA, tax NONE, no button or handicap bonus, friendlyPassOk=false")


def terminal_board_area_result(board: BoardState, *, komi: float,
                               rules: str | Mapping[str, object]) -> str:
    """Score two-pass completion under the explicit raw-board convention."""
    validate_board_area_rules(rules)
    if not math.isfinite(komi):
        raise ValueError("Komi must be finite")
    if board.consecutive_passes() < 2:
        raise ValueError("A board-area outcome requires two-pass termination; caps are not outcomes")
    prefix = board.clone()
    prefix.moves = prefix.moves[:-1]
    if prefix.has_two_pass_history():
        raise ValueError("History continues after an earlier two-pass termination")
    area = count_board_area(board.current_board())
    lead = area.black - area.white - komi
    if lead == 0:
        return "0"
    return f"{'B' if lead > 0 else 'W'}+{abs(lead):g}"
