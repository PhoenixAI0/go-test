"""Learner-turn training sources from GTP match SGFs, with full move history."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

from sgfmill import sgf

from .board import BoardState, coords_to_gtp
from .data_prep import state_record_from_board
from .game_result import parse_game_result
from .learner_data import _game_identity
from .rules import legality_superko


def sgf_training_records(
    data: bytes, *, learner_name: str, source_file: str,
    checkpoint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select only the named learner's pre-action states, never opponent turns.

    PB/PW matches are exact; a self-play SGF with the same learner on both sides
    contributes both colors. All intervening moves are replayed and retained.
    RE is source provenance, never a new training target. Void has no outcome.
    This adapter is for ordinary GTP matches, not SGF editing/encore workflows.
    """
    if not learner_name:
        raise ValueError("Specify an exact nonempty learner player name")
    game = sgf.Sgf_game.from_bytes(data)
    root = game.get_root()
    if root.has_property("GM") and root.get("GM") != 1:
        raise ValueError("The SGF does not describe Go")
    names = {color: root.get(key) if root.has_property(key) else ""
             for color, key in (("B", "PB"), ("W", "PW"))}
    learner_colors = {color for color, name in names.items() if name == learner_name}
    if not learner_colors:
        raise ValueError(f"Learner {learner_name!r} matches neither PB nor PW")
    if not root.has_property("RU") or not root.has_property("KM"):
        raise ValueError("Learner match SGFs must declare rules and komi")
    rules, komi = root.get("RU"), float(root.get("KM"))
    superko = legality_superko(rules)
    if not math.isfinite(komi):
        raise ValueError("SGF komi must be finite")
    raw_result = root.get("RE") if root.has_property("RE") else ""
    if raw_result.strip().upper() in {"", "VOID", "?"}:
        result, black_outcome = None, None
    else:
        result, black_outcome = parse_game_result(raw_result)
    comment = root.get("C") if root.has_property("C") else ""
    termination_match = re.search(r"(?:^|\s)termination=([a-z_]+)(?:\s|$)", comment)
    termination = termination_match.group(1) if termination_match else "unspecified"
    if termination == "move_cap" and result is not None:
        raise ValueError("A move-capped SGF cannot have an adjudicated outcome")
    size = game.get_size()
    matrix = [[0] * size for _ in range(size)]
    black, white, empty = root.get_setup_stones()
    if black & white or (black | white) & empty:
        raise ValueError("Conflicting SGF setup stones")
    for value, points in ((1, black), (2, white)):
        for row, col in points:
            matrix[size - 1 - row][col] = value
    initial_player = (root.get("PL").upper() if root.has_property("PL")
                      else "W" if root.has_property("HA") and root.get("HA") > 1 else "B")
    board = BoardState.from_matrix(size, matrix, next_player=initial_player)
    digest = hashlib.sha256(data).hexdigest()
    records = []
    for node in game.get_main_sequence():
        if node is not root and node.has_setup_stones():
            raise ValueError("Mid-game setup changes are not valid learner trajectories")
        color, coords = node.get_move()
        if color is None:
            continue
        player = color.upper()
        if player != board.next_player():
            raise ValueError("SGF move history does not alternate players")
        if board.consecutive_passes() >= 2:
            raise ValueError("SGF contains moves after two-pass termination")
        point = (size - 1 - coords[0], coords[1]) if coords is not None else None
        move = coords_to_gtp(point, size)
        if move not in board.legal_moves(player, superko=superko):
            raise ValueError(f"Illegal SGF action at ply {len(board.moves)}: {player} {move}")
        if player in learner_colors:
            record = {
                **state_record_from_board(board), "split": "train", "symmetry": 0,
                "rules": rules, "komi": komi, "learner_move": move,
                "source_type": "learner_gtp_match", "source_file": source_file,
                "teacher_labeled": False,
                "learner": {
                    "source_sgf_sha256": digest, "player_name": learner_name,
                    "player_color": player, "black_name": names["B"], "white_name": names["W"],
                    "ply": len(board.moves), "match_termination": termination,
                    "sgf_result": raw_result, "adjudicated_black_outcome": black_outcome,
                    "checkpoint": dict(checkpoint or {"available_at_export": False}),
                },
            }
            if result is not None:
                record["source_result"] = result
                record["source_outcome_source"] = "gtp_match_sgf_RE"
            records.append(record)
        board.apply_move(player, move, strict=True)
    if not records:
        raise ValueError("SGF has no recorded learner actions to export")
    identity = _game_identity(board, rules, komi)
    for record in records:
        record["game_id"] = identity
    return records
