from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"
_STONE = {"B": 1, "W": 2}
_EMPTY = 0


class GtpParseError(ValueError):
    pass


def gtp_to_coords(move: str, board_size: int) -> Optional[Tuple[int, int]]:
    move = move.strip().upper()
    if move == "PASS":
        return None
    if len(move) < 2:
        raise GtpParseError(f"Invalid GTP move: {move}")
    col_letter = move[0]
    if col_letter not in _GTP_COLUMNS:
        raise GtpParseError(f"Invalid column letter: {col_letter}")
    try:
        row = int(move[1:])
    except ValueError as exc:
        raise GtpParseError(f"Invalid row number: {move}") from exc
    col = _GTP_COLUMNS.index(col_letter)
    if row < 1 or row > board_size or col < 0 or col >= board_size:
        raise GtpParseError(f"Move out of bounds: {move}")
    return board_size - row, col


def coords_to_gtp(coords: Optional[Tuple[int, int]], board_size: int) -> str:
    if coords is None:
        return "PASS"
    row, col = coords
    if row < 0 or row >= board_size or col < 0 or col >= board_size:
        raise GtpParseError(f"Coords out of bounds: {coords}")
    col_letter = _GTP_COLUMNS[col]
    return f"{col_letter}{board_size - row}"


def _neighbors(row: int, col: int, size: int) -> List[Tuple[int, int]]:
    neighbors: List[Tuple[int, int]] = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row + 1 < size:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col + 1 < size:
        neighbors.append((row, col + 1))
    return neighbors


def _copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def _group_and_liberties(
    board: List[List[int]], start: Tuple[int, int]
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    size = len(board)
    color = board[start[0]][start[1]]
    group: Set[Tuple[int, int]] = set()
    liberties: Set[Tuple[int, int]] = set()
    stack = [start]
    while stack:
        row, col = stack.pop()
        if (row, col) in group:
            continue
        group.add((row, col))
        for n_row, n_col in _neighbors(row, col, size):
            value = board[n_row][n_col]
            if value == _EMPTY:
                liberties.add((n_row, n_col))
            elif value == color and (n_row, n_col) not in group:
                stack.append((n_row, n_col))
    return group, liberties


def _apply_strict_move(
    board: List[List[int]], color: int, coords: Tuple[int, int]
) -> List[List[int]]:
    size = len(board)
    row, col = coords
    if board[row][col] != _EMPTY:
        raise ValueError("Move occupied")
    new_board = _copy_board(board)
    new_board[row][col] = color
    opponent = _STONE["W"] if color == _STONE["B"] else _STONE["B"]
    visited: Set[Tuple[int, int]] = set()
    captured_any = False
    for n_row, n_col in _neighbors(row, col, size):
        if new_board[n_row][n_col] != opponent or (n_row, n_col) in visited:
            continue
        group, liberties = _group_and_liberties(new_board, (n_row, n_col))
        visited.update(group)
        if not liberties:
            captured_any = True
            for g_row, g_col in group:
                new_board[g_row][g_col] = _EMPTY
    group, liberties = _group_and_liberties(new_board, (row, col))
    if not liberties and not captured_any:
        raise ValueError("Move is suicide")
    return new_board


@dataclass
class BoardState:
    size: int
    moves: List[Tuple[str, Optional[Tuple[int, int]]]]

    @classmethod
    def empty(cls, size: int) -> "BoardState":
        return cls(size=size, moves=[])

    def _current_board(self) -> List[List[int]]:
        board = [[_EMPTY for _ in range(self.size)] for _ in range(self.size)]
        for index, (player, coords) in enumerate(self.moves, start=1):
            if coords is None:
                continue
            color = _STONE.get(player.upper())
            if color is None:
                raise ValueError(f"Unknown player at move {index}: {player}")
            try:
                board = _apply_strict_move(board, color, coords)
            except Exception as exc:
                raise ValueError(f"Illegal move at {index}: {player} {coords}") from exc
        return board

    def occupied(self) -> set[Tuple[int, int]]:
        board = self._current_board()
        stones = set()
        for row in range(self.size):
            for col in range(self.size):
                if board[row][col] != _EMPTY:
                    stones.add((row, col))
        return stones

    def next_player(self) -> str:
        if not self.moves:
            return "B"
        last_player = self.moves[-1][0].upper()
        if last_player == "B":
            return "W"
        if last_player == "W":
            return "B"
        black_moves = sum(1 for player, _ in self.moves if player.upper() == "B")
        white_moves = sum(1 for player, _ in self.moves if player.upper() == "W")
        return "B" if black_moves == white_moves else "W"

    def is_basic_legal(self, coords: Optional[Tuple[int, int]]) -> bool:
        if coords is None:
            return True
        row, col = coords
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        board = self._current_board()
        return board[row][col] == _EMPTY

    def _strict_board_state(
        self,
    ) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        board = [[_EMPTY for _ in range(self.size)] for _ in range(self.size)]
        previous: Optional[List[List[int]]] = None
        for player, coords in self.moves:
            previous = _copy_board(board)
            if coords is None:
                continue
            color = _STONE.get(player.upper())
            if color is None:
                raise ValueError(f"Unknown player: {player}")
            board = _apply_strict_move(board, color, coords)
        return board, previous

    def is_strict_legal(self, player: str, coords: Optional[Tuple[int, int]]) -> bool:
        if coords is None:
            return True
        color = _STONE.get(player.upper())
        if color is None:
            return False
        try:
            board, previous = self._strict_board_state()
        except Exception:
            return False
        row, col = coords
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if board[row][col] != _EMPTY:
            return False
        try:
            next_board = _apply_strict_move(board, color, coords)
        except Exception:
            return False
        if previous is not None and next_board == previous:
            return False
        return True

    def apply_move(self, player: str, move: str, strict: bool = False) -> None:
        player = player.strip().upper()
        if player not in _STONE:
            raise ValueError(f"Unknown player: {player}")
        coords = gtp_to_coords(move, self.size)
        if strict:
            is_legal = self.is_strict_legal(player, coords)
        else:
            is_legal = self.is_basic_legal(coords)
        if not is_legal:
            rule_label = "strict" if strict else "basic"
            raise ValueError(f"Illegal move for {rule_label} rules: {move}")
        self.moves.append((player, coords))

    def to_prompt(self) -> str:
        lines = [f"board_size: {self.size}"]
        for player, coords in self.moves:
            gtp_move = coords_to_gtp(coords, self.size)
            lines.append(f"{player}: {gtp_move}")
        lines.append("move:")
        return "\n".join(lines)
