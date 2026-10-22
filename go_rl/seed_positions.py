"""Training-only history validation for seeded self-play experiments."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .board import BoardState, coords_to_gtp
from .data_prep import board_from_example
from .experiment_utils import position_key
from .rules import normalize_rules
from .teacher_data import validate_history_for_rules


def replay_seed_board(start: BoardState, *, board_size: int, rules: Any) -> BoardState:
    """Rebuild a seed from empty, rejecting missing setup or terminal context."""
    if start.size != board_size:
        raise ValueError("Seed board size differs from the model")
    setup, moves = start.analysis_setup()
    if setup is not None and any(value for row in setup for value in row):
        raise ValueError("Seeded self-play currently rejects nonempty initial setup stones")
    replay = BoardState.empty(board_size)
    for player, coords in moves:
        if replay.consecutive_passes() >= 2:
            raise ValueError("Seed history continues after two-pass termination")
        if player != replay.next_player():
            raise ValueError("Seed must contain a complete alternating history from empty")
        replay.apply_move(player, coords_to_gtp(coords, board_size), strict=True)
    if replay.current_board() != start.current_board() or replay.next_player() != start.next_player():
        raise ValueError("Seed history cannot reconstruct the supplied position from empty")
    if replay.consecutive_passes() >= 2:
        raise ValueError("Seed is already terminal after two passes")
    validate_history_for_rules(replay, rules)
    return replay


def prepare_seed_records(
    records: Sequence[Mapping[str, Any]], *, board_size: int, rules: Any,
    komi: float, min_move: int = 250,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate explicit training membership and objective before seed sampling.

    Ordering and duplicate input rows cannot change the candidate list. Move
    labels, teacher values and game outcomes never determine seed eligibility.
    """
    if not isinstance(min_move, int) or isinstance(min_move, bool) or min_move < 0:
        raise ValueError("Minimum seed move must be a nonnegative integer")
    if not math.isfinite(komi):
        raise ValueError("Seed objective komi must be finite")
    target_rules = normalize_rules(rules)
    selected = {}
    rejected = {"before_minimum_move": 0, "initial_setup": 0, "terminal_history": 0}
    for row in records:
        if row.get("split") != "train":
            raise ValueError("Seed dataset must contain only explicitly pinned training records")
        if not isinstance(row.get("game_id"), str) or not row["game_id"]:
            raise ValueError("Seed records require game_id provenance")
        if "rules" not in row or "komi" not in row:
            raise ValueError("Seed records require explicit rules and komi")
        if normalize_rules(row["rules"]) != target_rules or float(row["komi"]) != komi:
            raise ValueError("Seed objective differs from the checkpoint/run rules or komi")
        if "moves" not in row or "initial_stones" not in row:
            raise ValueError("Seed records require complete initial stones and move history")
        if row.get("move_number") != len(row["moves"]):
            raise ValueError("Seed move_number differs from its complete history")
        if len(row["moves"]) < min_move:
            rejected["before_minimum_move"] += 1
            continue
        board = board_from_example(row)
        if row["initial_stones"]:
            rejected["initial_setup"] += 1
            continue
        if any(first[1] is None and second[1] is None for first, second in zip(board.moves, board.moves[1:])):
            rejected["terminal_history"] += 1
            continue
        replay_seed_board(board, board_size=board_size, rules=rules)
        selected[position_key(row)] = dict(row)
    if not selected:
        raise ValueError(f"No eligible training seed positions at the requested minimum move; rejected={rejected}")
    return [selected[key] for key in sorted(selected)], rejected


def sample_training_seed(records: Sequence[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Sample a source game uniformly, then one of its validated positions."""
    if not records:
        raise ValueError("Cannot sample an empty seed set")
    groups = defaultdict(list)
    for record in sorted(records, key=position_key):
        groups[record["game_id"]].append(record)
    games = sorted(groups)
    candidates = groups[games[rng.randrange(len(games))]]
    return candidates[rng.randrange(len(candidates))]
