"""Shared dataset isolation and inexpensive baselines for bounded experiments."""
from __future__ import annotations

import hashlib
from itertools import chain
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_prep import board_from_example
from .rules import normalize_rules


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def position_key(record: Mapping[str, Any]) -> str:
    return stable_digest([record.get("game_id"), record.get("move_number"), record.get("symmetry", 0),
                          record["prompt"], record.get("moves"), record.get("initial_stones"),
                          record.get("rules"), record.get("komi")])


def select_game_balanced(records: Sequence[dict[str, Any]], count: int, seed: int = 42) -> list[dict[str, Any]]:
    """Round-robin games deterministically, invariant to row order and duplicates."""
    if count < 1:
        raise ValueError("count must be positive")
    games: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        game_id = record.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("Benchmarks require a nonempty game_id on every record")
        games[game_id][position_key(record)] = record
    group_order = sorted(games, key=lambda game_id: (stable_digest([seed, "game", game_id]), game_id))
    ordered = {game_id: [games[game_id][key] for key in sorted(
        games[game_id], key=lambda key: (stable_digest([seed, "position", key]), key))]
        for game_id in group_order}
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < count:
        added = False
        for game_id in group_order:
            if offset < len(ordered[game_id]):
                selected.append(ordered[game_id][offset])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        offset += 1
    return selected


def select_game_balanced_indices(records: Sequence[dict[str, Any]], count: int, seed: int = 42) -> list[int]:
    """Exact stable selector above, retaining row indices and bounded candidates.

    At most ``count`` games and ``count`` candidate positions per game can be
    returned. A first streaming pass chooses those games; a second keeps their
    lowest-ranked unique positions. Duplicate position keys retain the last row,
    matching the original dict-based selector, without retaining full histories.
    """
    if count < 1:
        raise ValueError("count must be positive")
    games = set()
    for record in records:
        game_id = record.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("Benchmarks require a nonempty game_id on every record")
        games.add(game_id)
    game_order = sorted(games, key=lambda game_id: (stable_digest([seed, "game", game_id]), game_id))[:count]
    candidates: dict[str, dict[str, tuple[tuple[str, str], int]]] = {game_id: {} for game_id in game_order}
    for index, record in enumerate(records):
        selected = candidates.get(record["game_id"])
        if selected is None:
            continue
        key = position_key(record)
        rank = (stable_digest([seed, "position", key]), key)
        if key in selected or len(selected) < count:
            selected[key] = (rank, index)
        else:
            worst = max(selected, key=lambda candidate: selected[candidate][0])
            if rank < selected[worst][0]:
                del selected[worst]
                selected[key] = (rank, index)
    ordered = {game_id: [entry[1] for entry in sorted(candidates[game_id].values())]
               for game_id in game_order}
    result = []
    for offset in range(count):
        added = False
        for game_id in game_order:
            if offset < len(ordered[game_id]):
                result.append(ordered[game_id][offset])
                added = True
                if len(result) == count:
                    return result
        if not added:
            break
    return result


def read_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_identity(record: Mapping[str, Any]) -> str:
    """Canonical visible state/objective; intentionally ignores different history.

    This is a stronger isolation check than exact Markov-state equality: even
    an opening position reached in another game or orientation is held apart.
    """
    board = board_from_example(record)
    matrix = board.current_board()
    size = board.size
    variants = []
    for symmetry in range(8):
        transformed = [[0] * size for _ in range(size)]
        for row in range(size):
            for col in range(size):
                out_row, out_col = row, size - 1 - col if symmetry >= 4 else col
                for _ in range(symmetry % 4):
                    out_row, out_col = out_col, size - 1 - out_row
                transformed[out_row][out_col] = matrix[row][col]
        variants.append("".join(str(value) for cells in transformed for value in cells))
    komi = float(record["komi"])
    if not math.isfinite(komi):
        raise ValueError("Komi must be finite")
    return json.dumps([size, min(variants), board.next_player(), normalize_rules(record["rules"]), komi], sort_keys=True)


def validate_splits(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    if not train or not validation:
        raise ValueError("Both training and validation need at least one record")
    for partition, rows, allowed in (("Training", train, {"train"}),
                                      ("Evaluation", validation, {"validation", "test"})):
        for row in rows:
            pinned = row.get("split")
            if pinned is not None and pinned != "" and (not isinstance(pinned, str) or pinned not in allowed):
                raise ValueError(f"{partition} contains a record pinned to split {pinned!r}")
    if any(not isinstance(row.get("game_id"), str) or not row["game_id"] for row in chain(train, validation)):
        raise ValueError("Every record needs a nonempty game_id")
    shared = {row["game_id"] for row in train} & {row["game_id"] for row in validation}
    if shared:
        raise ValueError(f"Training/validation game leakage: {len(shared)} shared games")
    # Keep only held-out identities. Training may be a very large indexed JSONL
    # sequence, so neither its records nor all of its state strings should live
    # in memory. Identity retains player, exact rules/komi, and D4 equivalence.
    validation_states = {state_identity(row) for row in validation}
    states = {identity for row in train if (identity := state_identity(row)) in validation_states}
    if states:
        raise ValueError(f"Training/validation state leakage: {len(states)} shared states up to symmetry")


def policy_baselines(records: Sequence[Mapping[str, Any]], training: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Uniform legal-action and training-frequency baselines, without a model.

    Inputs use prepared records: legal (bool action mask), target (probabilities),
    and action (teacher top move). Frequency fitting never consumes eval labels.
    """
    if not records:
        raise ValueError("Baselines need at least one record")
    size = len(records[0]["legal"])
    frequencies = [1e-6] * size
    for row in training or []:
        for index, probability in enumerate(row["target"]):
            frequencies[index] += float(probability)
    metrics = {"uniform": {"policy_loss": 0., "top1_expected": 0., "top5_expected": 0.},
               "training_frequency": {"policy_loss": 0., "top1": 0., "top5": 0.}}
    for row in records:
        legal = [index for index, allowed in enumerate(row["legal"]) if allowed]
        if not legal or len(row["target"]) != size or len(row["legal"]) != size:
            raise ValueError("Baseline records need consistent nonempty legal masks and targets")
        target = row["target"]
        if any(not math.isfinite(float(p)) or p < 0 for p in target) or not math.isclose(sum(target), 1., abs_tol=1e-4):
            raise ValueError("Target policy must be a finite probability distribution")
        if any(p > 0 and not row["legal"][index] for index, p in enumerate(target)):
            raise ValueError("Target policy assigns mass to illegal actions")
        if row["action"] not in legal:
            raise ValueError("Reference action is illegal")
        metrics["uniform"]["policy_loss"] += math.log(len(legal))
        metrics["uniform"]["top1_expected"] += 1 / len(legal)
        metrics["uniform"]["top5_expected"] += min(5, len(legal)) / len(legal)
        total = sum(frequencies[index] for index in legal)
        ordered = sorted(legal, key=lambda index: (-frequencies[index], index))
        metrics["training_frequency"]["policy_loss"] -= sum(
            float(target[index]) * math.log(frequencies[index] / total) for index in legal)
        metrics["training_frequency"]["top1"] += ordered[0] == row["action"]
        metrics["training_frequency"]["top5"] += row["action"] in ordered[:5]
    return {key: {metric: value / len(records) for metric, value in values.items()} for key, values in metrics.items()}
