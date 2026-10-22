"""Replay learner episodes into unlabelled training positions for reanalysis.

The learner chooses every trajectory action. Exporting a position does not make
its sampled action an expert target, nor a truncated episode a scored game.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .board import BoardState
from .data_prep import board_from_example, state_record_from_board, transform_example
from .experiment_utils import stable_digest, state_identity
from .game_result import parse_game_result
from .rules import legality_superko, normalize_rules
from .seed_positions import replay_seed_board


def _game_identity(board: BoardState, rules: Any, komi: float) -> str:
    """Group equivalent complete trajectories, independent of run or orientation."""
    record = state_record_from_board(board)
    variants = []
    for symmetry in range(8):
        transformed = transform_example(record, symmetry)
        variants.append(stable_digest([transformed["initial_stones"], transformed["moves"]]))
    return "learner-" + stable_digest([board.size, min(variants), normalize_rules(rules), komi])


def episode_training_records(
    episode: Mapping[str, Any], run: Mapping[str, Any], *,
    source_run_id: str, episode_row: int,
    checkpoint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Export all pre-action states with rules, history and learner provenance.

    ``run`` is the saved ``rl_run.json`` and ``episode`` is one row of
    ``episodes.jsonl`` produced by run_policy_rl. No model or engine is loaded.
    Source actions are deliberately named ``learner_move``, never ``response``.
    Records are training-only; a later teacher annotation supplies supervision.
    """
    size = run.get("board_size")
    if not isinstance(size, int) or isinstance(size, bool) or not 2 <= size <= 25:
        raise ValueError("Run must specify an integer board_size in [2,25]")
    if "rules" not in run or "komi" not in run:
        raise ValueError("Run needs explicit rules and komi")
    rules, komi = run["rules"], float(run["komi"])
    superko = legality_superko(rules)
    if not math.isfinite(komi):
        raise ValueError("Komi must be finite")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("source_run_id must be nonempty")
    if not isinstance(episode_row, int) or episode_row < 0:
        raise ValueError("episode_row must be a nonnegative integer")
    moves = episode.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("Episode must contain a nonempty complete move list")
    if episode.get("plies") != len(moves):
        raise ValueError("Episode plies do not match its move history")
    terminated, truncated = episode.get("terminated"), episode.get("truncated")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool) or terminated == truncated:
        raise ValueError("Episode must be either terminated or truncated")
    if truncated and (episode.get("black_outcome") is not None or episode.get("final_score") is not None):
        raise ValueError("A truncated episode cannot have a scored outcome")
    if terminated:
        outcome = episode.get("black_outcome")
        if (isinstance(outcome, bool) or outcome not in (-1., 0., 1.)
                or not isinstance(episode.get("final_score"), str)
                or episode.get("outcome_source") != "katago_gtp_final_score"):
            raise ValueError("Terminated episodes require an explicit referee outcome")
        _, parsed_outcome = parse_game_result(episode["final_score"])
        if parsed_outcome != outcome:
            raise ValueError("Referee score and recorded black_outcome disagree")
    temperature = float(episode.get("behavior_temperature", run.get("temperature", 1.)))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Behavior temperature must be finite and positive")
    start_position = episode.get("start_position")
    board = (replay_seed_board(board_from_example(start_position), board_size=size, rules=rules)
             if start_position is not None else BoardState.empty(size))
    if "starting_plies" in episode and episode["starting_plies"] != len(board.moves):
        raise ValueError("Episode starting_plies differs from its seed history")
    records = []
    for ply, item in enumerate(moves):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Every episode move must contain player and GTP coordinate")
        player, move = item
        if player != board.next_player() or not isinstance(move, str):
            raise ValueError("Episode players must alternate from the supplied start")
        if board.consecutive_passes() >= 2:
            raise ValueError("Episode contains moves after two-pass termination")
        if move.upper() not in board.legal_moves(player, superko=superko):
            raise ValueError(f"Illegal learner action at ply {ply}: {player} {move}")
        records.append({
            **state_record_from_board(board), "rules": rules, "komi": komi,
            "split": "train", "symmetry": 0, "learner_move": move.upper(),
            "source_type": "learner_selfplay", "teacher_labeled": False,
            "learner": {
                "source_run_id": source_run_id, "run_seed": run.get("seed"),
                "update": episode.get("update"), "episode": episode.get("episode"),
                "episode_row": episode_row, "ply": len(board.moves), "rollout_step": ply,
                "seed_game_id": episode.get("seed_game_id"),
                "seed_position_key": episode.get("seed_position_key"),
                "behavior_temperature": temperature,
                "episode_terminated": terminated, "episode_truncated": truncated,
                "starting_checkpoint": dict(checkpoint or {"path": run.get("model")}),
                "behavior_policy": "starting_checkpoint plus all PPO updates before the recorded update",
            },
        })
        board.apply_move(player, move, strict=True)
    if terminated != (board.consecutive_passes() >= 2):
        raise ValueError("Episode boundary disagrees with two-pass termination")
    game_id = _game_identity(board, rules, komi)
    for record in records:
        record["game_id"] = game_id
        if terminated:
            # Provenance only: target rules/komi may change during reanalysis.
            record["source_result"] = episode["final_score"]
            record["source_outcome_source"] = episode["outcome_source"]
    return records


def deduplicate_training_records(
    records: Iterable[Mapping[str, Any]], *,
    excluded_records: Iterable[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate D4-equivalent visible states without selecting on move quality.

    The identity includes side-to-play, rules and komi. Different histories with
    the same visible state are intentionally collapsed, a conservative diversity
    filter that is stronger than equality of the full Markov state. Exclusions
    use only these state fields, never held-out responses or teacher values.
    """
    excluded = {state_identity(row) for row in excluded_records}
    seen: set[str] = set()
    kept = []
    counts = {"input_positions": 0, "duplicate_states": 0, "excluded_states": 0,
              "exported_positions": 0}
    for row in records:
        counts["input_positions"] += 1
        if row.get("split") != "train":
            raise ValueError("Learner refresh records must be explicitly training-only")
        identity = state_identity(row)
        if identity in excluded:
            counts["excluded_states"] += 1
            continue
        if identity in seen:
            counts["duplicate_states"] += 1
            continue
        seen.add(identity)
        kept.append(dict(row))
    counts["exported_positions"] = len(kept)
    return kept, counts
