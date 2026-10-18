from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Set, Tuple

_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"
_STONE = {"B": 1, "W": 2}
_EMPTY = 0
_STONE_TO_CHAR = {_EMPTY: ".", _STONE["B"]: "X", _STONE["W"]: "O"}


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
    moves: List[Tuple[str, Optional[Tuple[int, int]]]] = field(default_factory=list)
    board_snapshot: Optional[List[List[int]]] = None
    next_player_override: Optional[str] = None
    previous_board_snapshot: Optional[List[List[int]]] = None

    @classmethod
    def empty(cls, size: int) -> "BoardState":
        return cls(size=size, moves=[])

    @classmethod
    def from_matrix(
        cls,
        size: int,
        board: List[List[int]],
        next_player: str,
        previous_board: Optional[List[List[int]]] = None,
    ) -> "BoardState":
        normalized = next_player.strip().upper()
        if normalized not in _STONE:
            raise ValueError(f"Unknown player: {next_player}")
        if len(board) != size or any(len(row) != size for row in board):
            raise ValueError("Board matrix shape does not match board size")
        if previous_board is not None:
            if len(previous_board) != size or any(len(row) != size for row in previous_board):
                raise ValueError("Previous board matrix shape does not match board size")
        return cls(
            size=size,
            moves=[],
            board_snapshot=_copy_board(board),
            next_player_override=normalized,
            previous_board_snapshot=_copy_board(previous_board) if previous_board is not None else None,
        )

    def has_snapshot(self) -> bool:
        return self.board_snapshot is not None

    def clone(self) -> "BoardState":
        if self.board_snapshot is not None:
            return BoardState.from_matrix(
                size=self.size,
                board=self.board_snapshot,
                next_player=self.next_player(),
                previous_board=self.previous_board_snapshot,
            )
        return BoardState(size=self.size, moves=list(self.moves))

    def _replay(self) -> Iterator[List[List[int]]]:
        """Yield the empty board, then a fresh board after each recorded move."""
        board = [[_EMPTY for _ in range(self.size)] for _ in range(self.size)]
        yield board
        for index, (player, coords) in enumerate(self.moves, start=1):
            if coords is not None:
                color = _STONE.get(player.upper())
                if color is None:
                    raise ValueError(f"Unknown player at move {index}: {player}")
                try:
                    board = _apply_strict_move(board, color, coords)
                except ValueError as exc:
                    raise ValueError(f"Illegal move at {index}: {player} {coords}") from exc
            yield board

    def _current_board(self) -> List[List[int]]:
        if self.board_snapshot is not None:
            return _copy_board(self.board_snapshot)
        *_, board = self._replay()
        return board

    def position_history(self) -> List[List[List[int]]]:
        """All known whole-board positions so far, oldest first (for superko).

        Snapshot boards only know their current and previous positions.
        """
        if self.board_snapshot is not None:
            positions = [self.previous_board_snapshot, self.board_snapshot]
            return [_copy_board(board) for board in positions if board is not None]
        return list(self._replay())

    def current_board(self) -> List[List[int]]:
        return self._current_board()

    def previous_board(self) -> Optional[List[List[int]]]:
        if self.previous_board_snapshot is None:
            return None
        return _copy_board(self.previous_board_snapshot)

    def occupied(self) -> set[Tuple[int, int]]:
        board = self._current_board()
        stones = set()
        for row in range(self.size):
            for col in range(self.size):
                if board[row][col] != _EMPTY:
                    stones.add((row, col))
        return stones

    def stones(self) -> List[Tuple[str, Tuple[int, int]]]:
        stones: List[Tuple[str, Tuple[int, int]]] = []
        board = self._current_board()
        for row in range(self.size):
            for col in range(self.size):
                value = board[row][col]
                if value == _STONE["B"]:
                    stones.append(("B", (row, col)))
                elif value == _STONE["W"]:
                    stones.append(("W", (row, col)))
        return stones

    def next_player(self) -> str:
        if self.next_player_override is not None:
            return self.next_player_override
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
        if self.board_snapshot is not None:
            return _copy_board(self.board_snapshot), self.previous_board()
        positions = list(self._replay())
        # The ko comparison point is the position before the last move (or pass).
        return positions[-1], positions[-2] if len(positions) > 1 else None

    def is_strict_legal(self, player: str, coords: Optional[Tuple[int, int]]) -> bool:
        if coords is None:
            return True
        color = _STONE.get(player.upper())
        if color is None:
            return False
        try:
            board, previous = self._strict_board_state()
        except ValueError:
            return False
        row, col = coords
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if board[row][col] != _EMPTY:
            return False
        try:
            next_board = _apply_strict_move(board, color, coords)
        except ValueError:
            return False
        if previous is not None and next_board == previous:
            return False
        return True

    def apply_move(
        self,
        player: str,
        move: str | Tuple[int, int] | None,
        strict: bool = False,
    ) -> None:
        player = player.strip().upper()
        if player not in _STONE:
            raise ValueError(f"Unknown player: {player}")
        if move is None:
            coords = None
            move_label = "PASS"
        elif isinstance(move, tuple):
            coords = move
            move_label = str(move)
        else:
            coords = gtp_to_coords(move, self.size)
            move_label = move
        if strict:
            is_legal = self.is_strict_legal(player, coords)
        else:
            is_legal = self.is_basic_legal(coords)
        if not is_legal:
            rule_label = "strict" if strict else "basic"
            raise ValueError(f"Illegal move for {rule_label} rules: {move_label}")
        if self.board_snapshot is not None:
            current = self._current_board()
            self.previous_board_snapshot = _copy_board(current)
            if coords is not None:
                color = _STONE[player]
                current = _apply_strict_move(current, color, coords)
            self.board_snapshot = current
            self.next_player_override = "W" if player == "B" else "B"
            return
        self.moves.append((player, coords))

    def to_prompt(self, player: Optional[str] = None) -> str:
        to_play = player.strip().upper() if player is not None else self.next_player()
        if to_play not in _STONE:
            raise ValueError(f"Unknown player: {player}")
        board = self._current_board()
        lines = [f"board_size: {self.size}", f"to_play: {to_play}", "board:"]
        for row in board:
            row_text = "".join(_STONE_TO_CHAR[value] for value in row)
            lines.append(row_text)
        lines.append("move:")
        return "\n".join(lines)
