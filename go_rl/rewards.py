from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .board import BoardState, gtp_to_coords
from .config import RewardConfig
from .rules import BoardHistory, LegalityLevel, validate_legality


@dataclass
class RewardBreakdown:
    winrate: float
    legality: float
    length: float
    outcome: float

    @property
    def total(self) -> float:
        return self.winrate + self.legality + self.length + self.outcome


def winrate_reward(delta: float, config: RewardConfig) -> float:
    return delta * config.winrate_scale


def legality_reward(is_legal: bool, config: RewardConfig) -> float:
    return config.legal_bonus if is_legal else config.illegal_penalty


def length_reward(output: str, config: RewardConfig) -> float:
    trimmed = output.strip()
    if trimmed == "PASS" or len(trimmed) <= 3:
        return 0.0
    return config.length_penalty


def outcome_reward(result: Optional[float], config: RewardConfig) -> float:
    if result is None:
        return 0.0
    return config.full_game_bonus * result


def resolve_legality_level(config: RewardConfig) -> LegalityLevel:
    if config.legality_level:
        try:
            return LegalityLevel(config.legality_level.lower())
        except ValueError as exc:
            raise ValueError(
                f"Unknown legality_level: {config.legality_level}"
            ) from exc
    return LegalityLevel.KO if config.strict_legality else LegalityLevel.BASIC


def _strict_state_with_history(
    board: BoardState,
) -> tuple[list[list[int]], Optional[list[list[int]]], BoardHistory]:
    # Import board internals locally so reward scoring only depends on them when needed.
    from .board import _EMPTY, _STONE, _apply_strict_move, _copy_board

    if board.has_snapshot():
        current = board.current_board()
        history = BoardHistory()
        history.add(current)
        return current, board.previous_board(), history
    current = [[_EMPTY for _ in range(board.size)] for _ in range(board.size)]
    history = BoardHistory()
    history.add(current)
    previous: Optional[list[list[int]]] = None
    for player, coords in board.moves:
        previous = _copy_board(current)
        if coords is None:
            continue
        color = _STONE.get(player.upper())
        if color is None:
            raise ValueError(f"Unknown player: {player}")
        current = _apply_strict_move(current, color, coords)
        history.add(current)
    return current, previous, history


def is_legal_coords(
    board: BoardState,
    player: str,
    coords: Optional[tuple[int, int]],
    config: RewardConfig,
) -> bool:
    if coords is None:
        return True
    level = resolve_legality_level(config)
    if level == LegalityLevel.BASIC:
        return board.is_basic_legal(coords)
    if level == LegalityLevel.KO:
        return board.is_strict_legal(player, coords)
    try:
        current, previous, history = _strict_state_with_history(board)
        color = _STONE.get(player.upper())
        if color is None:
            return False
        next_board = _apply_strict_move(current, color, coords)
        is_legal, _ = validate_legality(
            next_board, previous, history, LegalityLevel.SUPERKO
        )
        return is_legal
    except ValueError:
        return False


def is_legal_move(
    board: BoardState,
    player: str,
    move: str,
    config: RewardConfig,
) -> bool:
    try:
        coords = gtp_to_coords(move, board.size)
    except ValueError:
        return False
    return is_legal_coords(board, player, coords, config)


def evaluate_move(
    board: BoardState,
    move: str,
    winrate_delta: float,
    config: RewardConfig,
    game_result: Optional[float] = None,
    player: str = "B",
) -> RewardBreakdown:
    try:
        coords = gtp_to_coords(move, board.size)
        is_legal = is_legal_coords(board, player, coords, config)
    except ValueError:
        is_legal = False
    return RewardBreakdown(
        winrate=winrate_reward(winrate_delta, config),
        legality=legality_reward(is_legal, config),
        length=length_reward(move, config),
        outcome=outcome_reward(game_result, config),
    )
