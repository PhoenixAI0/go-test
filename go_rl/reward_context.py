"""Reconstruct the actual task a generated move is being rewarded for."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Mapping

from .board import BoardState
from .config import RewardConfig
from .data_prep import board_from_example
from .rules import legality_superko, normalize_rules
from .shared_types import board_key


@dataclass
class RewardPosition:
    board: BoardState
    reward_config: RewardConfig
    analysis_options: dict[str, object]

    def key(self) -> tuple:
        # Same stones/history under different komi/rules are different tasks.
        return (board_key(self.board), json.dumps(self.analysis_options, sort_keys=True))


def reward_position(
    prompt: str, columns: Mapping[str, object], index: int,
    count: int, reward_config: RewardConfig,
    default_rules: str | dict[str, object] | None = None,
    default_komi: float | None = None,
) -> RewardPosition:
    """Read one TRL example's raw board/history and per-example rule metadata.

    Dataset errors fail loudly; they are not the model's illegal-action penalty.
    Legacy prompt-only datasets keep configured defaults but cannot recreate
    superko history that was never stored.
    """
    example: dict[str, object] = {"prompt": prompt}
    for key in ("board_size", "moves", "initial_stones", "initial_player", "to_play", "rules", "komi"):
        if key not in columns:
            continue
        values = columns[key]
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError(f"Reward metadata column {key} must contain {count} rows")
        if values[index] is not None:
            example[key] = values[index]
    if "rules" not in example and default_rules is not None:
        example["rules"] = default_rules
    if "komi" not in example and default_komi is not None:
        example["komi"] = default_komi
    options: dict[str, object] = {}
    if "rules" in example:
        rules = normalize_rules(example["rules"])
        superko = legality_superko(rules)
        reward_config = replace(reward_config, strict_legality=True,
                                legality_level="superko" if superko else "ko")
        options["rules"] = example["rules"]
    if "komi" in example:
        komi = float(example["komi"])
        if not math.isfinite(komi):
            raise ValueError("Reward komi must be finite")
        options["komi"] = komi
    board = board_from_example(example)
    if ("rules" in example and normalize_rules(example["rules"])["scoring"] == "AREA"
            and board.has_two_pass_history()):
        raise ValueError("Area-scoring reward positions must precede two-pass termination")
    return RewardPosition(board, reward_config, options)
