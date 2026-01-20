from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .board import gtp_to_coords, BoardState
from .config import RewardConfig


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
        if config.strict_legality:
            is_legal = board.is_strict_legal(player, coords)
        else:
            is_legal = board.is_basic_legal(coords)
    except Exception:
        is_legal = False
    return RewardBreakdown(
        winrate=winrate_reward(winrate_delta, config),
        legality=legality_reward(is_legal, config),
        length=length_reward(move, config),
        outcome=outcome_reward(game_result, config),
    )
