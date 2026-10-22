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
    """A setup position and subsequent moves, with cached immutable replay states.

    ``board_snapshot`` is the initial setup, not the latest position. Always use
    ``current_board()`` to read current stones. Keeping setup and moves separate
    lets KataGo receive the same ko/pass history that our legality checks use.
    Returned matrices are copies; cached matrices are never modified in place.
    """

    size: int
    moves: List[Tuple[str, Optional[Tuple[int, int]]]] = field(default_factory=list)
    board_snapshot: Optional[List[List[int]]] = None
    next_player_override: Optional[str] = None
    previous_board_snapshot: Optional[List[List[int]]] = None
    _cached_setup: object = field(default=None, init=False, repr=False, compare=False)
    _cached_moves: tuple = field(default=(), init=False, repr=False, compare=False)
    _cached_positions: List[List[List[int]]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or not 1 <= self.size <= len(_GTP_COLUMNS):
            raise ValueError("Board size must be an integer between 1 and 19")
        for label, matrix in (
            ("Board", self.board_snapshot),
            ("Previous board", self.previous_board_snapshot),
        ):
            if matrix is not None:
                if len(matrix) != self.size or any(len(row) != self.size for row in matrix):
                    raise ValueError(f"{label} matrix shape does not match board size")
                if any(value not in (0, 1, 2) for row in matrix for value in row):
                    raise ValueError(f"{label} matrix contains invalid stone values")
        if self.next_player_override is not None:
            normalized = self.next_player_override.strip().upper()
            if normalized not in _STONE:
                raise ValueError(f"Unknown player: {self.next_player_override}")
            self.next_player_override = normalized

    @classmethod
    def empty(cls, size: int) -> "BoardState":
        return cls(size=size)

    @classmethod
    def from_matrix(
        cls,
        size: int,
        board: List[List[int]],
        next_player: str,
        previous_board: Optional[List[List[int]]] = None,
    ) -> "BoardState":
        return cls(
            size=size,
            board_snapshot=_copy_board(board),
            next_player_override=next_player,
            previous_board_snapshot=_copy_board(previous_board) if previous_board is not None else None,
        )

    def has_snapshot(self) -> bool:
        return self.board_snapshot is not None

    def clone(self) -> "BoardState":
        clone = BoardState(
            size=self.size,
            moves=list(self.moves),
            board_snapshot=_copy_board(self.board_snapshot) if self.board_snapshot is not None else None,
            next_player_override=self.next_player_override,
            previous_board_snapshot=(
                _copy_board(self.previous_board_snapshot)
                if self.previous_board_snapshot is not None else None
            ),
        )
        # Matrices in this cache are immutable by convention. Branches only add
        # new matrices; sharing old positions avoids O(game length * area) copies.
        clone._cached_setup = self._cached_setup
        clone._cached_moves = self._cached_moves
        clone._cached_positions = list(self._cached_positions)
        return clone

    def _positions(self) -> List[List[List[int]]]:
        setup = (
            self.size,
            tuple(tuple(row) for row in self.board_snapshot)
            if self.board_snapshot is not None else None,
        )
        moves = tuple(self.moves)
        if self._cached_positions and self._cached_setup == setup and self._cached_moves == moves:
            return self._cached_positions
        if (self._cached_positions and self._cached_setup == setup
                and moves[:len(self._cached_moves)] == self._cached_moves):
            positions = list(self._cached_positions)
            start = len(self._cached_moves)
        else:
            initial = (_copy_board(self.board_snapshot) if self.board_snapshot is not None
                       else [[_EMPTY] * self.size for _ in range(self.size)])
            positions = [initial]
            start = 0
        board = positions[-1]
        for index in range(start, len(moves)):
            player, coords = moves[index]
            color = _STONE.get(player.upper())
            if color is None:
                raise ValueError(f"Unknown player at move {index + 1}: {player}")
            if coords is not None:
                row, col = coords
                if not 0 <= row < self.size or not 0 <= col < self.size:
                    raise ValueError(f"Move out of bounds at {index + 1}: {coords}")
                try:
                    board = _apply_strict_move(board, color, coords)
                except ValueError as exc:
                    raise ValueError(f"Illegal move at {index + 1}: {player} {coords}") from exc
            positions.append(board)
        self._cached_setup = setup
        self._cached_moves = moves
        self._cached_positions = positions
        return positions

    def _replay(self) -> Iterator[List[List[int]]]:
        """Yield independent copies of the setup and every subsequent position."""
        for board in self._positions():
            yield _copy_board(board)

    def _current_board(self) -> List[List[int]]:
        return _copy_board(self._positions()[-1])

    def position_history(self) -> List[List[List[int]]]:
        """All known positions, including setup context, oldest first."""
        positions = self._positions()
        if self.previous_board_snapshot is not None:
            positions = [self.previous_board_snapshot, *positions]
        return [_copy_board(board) for board in positions]

    def current_board(self) -> List[List[int]]:
        return self._current_board()

    def previous_board(self) -> Optional[List[List[int]]]:
        _, previous = self._strict_board_state()
        return _copy_board(previous) if previous is not None else None

    def occupied(self) -> set[Tuple[int, int]]:
        return {(row, col) for row, cells in enumerate(self._positions()[-1])
                for col, value in enumerate(cells) if value != _EMPTY}

    def stones(self) -> List[Tuple[str, Tuple[int, int]]]:
        return [("B" if value == _STONE["B"] else "W", (row, col))
                for row, cells in enumerate(self._positions()[-1])
                for col, value in enumerate(cells) if value != _EMPTY]

    def next_player(self) -> str:
        if self.moves:
            last_player = self.moves[-1][0].strip().upper()
            if last_player not in _STONE:
                raise ValueError(f"Unknown player: {self.moves[-1][0]}")
            return "W" if last_player == "B" else "B"
        return self.next_player_override or "B"

    def consecutive_passes(self) -> int:
        count = 0
        for _, coords in reversed(self.moves):
            if coords is not None:
                return count
            count += 1
        if self.board_snapshot is not None and self.previous_board_snapshot == self.board_snapshot:
            count += 1
        return count

    def has_two_pass_history(self) -> bool:
        """Whether known history ever reaches two consecutive passes.

        This includes a pass identifiable immediately before a snapshot. It is
        a history fact; whether it ends a game depends on the experiment/rules.
        """
        passes = int(self.board_snapshot is not None
                     and self.previous_board_snapshot == self.board_snapshot)
        for _, coords in self.moves:
            passes = passes + 1 if coords is None else 0
            if passes >= 2:
                return True
        return False

    def is_basic_legal(self, coords: Optional[Tuple[int, int]]) -> bool:
        if coords is None:
            return True
        row, col = coords
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return False
        return self._positions()[-1][row][col] == _EMPTY

    def _strict_board_state(
        self,
    ) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        positions = self._positions()
        previous = positions[-2] if len(positions) > 1 else self.previous_board_snapshot
        return positions[-1], previous

    def is_strict_legal(self, player: str, coords: Optional[Tuple[int, int]]) -> bool:
        color = _STONE.get(player.strip().upper())
        if color is None:
            return False
        if coords is None:
            return True
        row, col = coords
        if not 0 <= row < self.size or not 0 <= col < self.size:
            return False
        try:
            board, previous = self._strict_board_state()
            next_board = _apply_strict_move(board, color, coords)
        except ValueError:
            return False
        return previous is None or next_board != previous

    def legal_moves(
        self,
        player: Optional[str] = None,
        *,
        superko: bool = False,
        include_pass: bool = True,
    ) -> List[str]:
        """List legal GTP actions in board order, optionally with positional superko.

        A snapshot can only enforce repetition against its known history. Use
        the original setup and full move prefix when strict superko is needed.
        """
        to_play = player.strip().upper() if player is not None else self.next_player()
        if to_play not in _STONE:
            raise ValueError(f"Unknown player: {player}")
        board, previous = self._strict_board_state()
        history = {tuple(tuple(row) for row in position) for position in self.position_history()} if superko else set()
        legal = []
        for row in range(self.size):
            for col in range(self.size):
                if board[row][col] != _EMPTY:
                    continue
                try:
                    candidate = _apply_strict_move(board, _STONE[to_play], (row, col))
                except ValueError:
                    continue
                if previous is not None and candidate == previous:
                    continue
                if superko and tuple(tuple(cells) for cells in candidate) in history:
                    continue
                legal.append(coords_to_gtp((row, col), self.size))
        if include_pass:
            legal.append("PASS")
        return legal

    def analysis_setup(self) -> Tuple[Optional[List[List[int]]], List[Tuple[str, Optional[Tuple[int, int]]]]]:
        """Return KataGo's initial stones and moves without losing ko context.

        A legacy snapshot may provide only the immediately preceding board.
        Recover its last move (including captures) when this transition is
        verifiable. Reject inconsistent context instead of silently dropping it.
        """
        moves = list(self.moves)
        initial = self.board_snapshot
        previous = self.previous_board_snapshot
        if initial is None or previous is None:
            return _copy_board(initial) if initial is not None else None, moves
        to_play = self.next_player_override or (moves[0][0].upper() if moves else "B")
        last_player = "W" if to_play == "B" else "B"
        if initial == previous:
            return _copy_board(previous), [(last_player, None), *moves]
        color = _STONE[last_player]
        additions = [(row, col) for row in range(self.size) for col in range(self.size)
                     if initial[row][col] == color and previous[row][col] != color]
        if len(additions) == 1:
            try:
                reconstructed = _apply_strict_move(previous, color, additions[0])
            except ValueError:
                reconstructed = None
            if reconstructed == initial:
                return _copy_board(previous), [(last_player, additions[0]), *moves]
        raise ValueError("Snapshot previous_board is not a legal predecessor; provide the original setup and move history")

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
            coords, move_label = None, "PASS"
        elif isinstance(move, tuple):
            coords, move_label = move, str(move)
        else:
            coords, move_label = gtp_to_coords(move, self.size), move
        is_legal = self.is_strict_legal(player, coords) if strict else self.is_basic_legal(coords)
        if not is_legal:
            rule_label = "strict" if strict else "basic"
            raise ValueError(f"Illegal move for {rule_label} rules: {move_label}")
        self.moves.append((player, coords))

    def to_prompt(self, player: Optional[str] = None) -> str:
        to_play = player.strip().upper() if player is not None else self.next_player()
        if to_play not in _STONE:
            raise ValueError(f"Unknown player: {player}")
        board = self._positions()[-1]
        lines = [f"board_size: {self.size}", f"to_play: {to_play}", "board:"]
        for row in board:
            lines.append("".join(_STONE_TO_CHAR[value] for value in row))
        lines.append("move:")
        return "\n".join(lines)
