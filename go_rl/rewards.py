from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .board import BoardState, gtp_to_coords
from .config import RewardConfig


class LegalityLevel(Enum):
    BASIC = "basic"  # Occupancy only.
    KO = "ko"  # Adds suicide and simple ko.
    SUPERKO = "superko"  # Adds positional superko.


@dataclass
class RewardBreakdown:
    winrate: float
    legality: float
    outcome: float

    @property
    def total(self) -> float:
        return self.winrate + self.legality + self.outcome


def winrate_reward(delta: float, config: RewardConfig) -> float:
    return delta * config.winrate_scale


def legality_reward(is_legal: bool, config: RewardConfig) -> float:
    return config.legal_bonus if is_legal else config.illegal_penalty


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
    if not board.is_strict_legal(player, coords):
        return False
    if level == LegalityLevel.KO:
        return True
    # Positional superko: the move may not recreate any earlier position.
    after = board.clone()
    after.apply_move(player, coords, strict=True)
    return after.current_board() not in board.position_history()


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
        outcome=outcome_reward(game_result, config),
    )
