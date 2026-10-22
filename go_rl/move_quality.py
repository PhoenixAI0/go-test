"""Finite-budget, paired engine estimates of chosen-move quality.

Every legal chosen child is analyzed explicitly, even when the root search
never explored that action. Differences from the reference are estimates, not
exact regret or game outcomes. Statistical intervals resample source games.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .board import BoardState
from .data_prep import board_from_example, state_record_from_board
from .experiment_utils import position_key, stable_digest
from .rules import legality_superko


def terminal_history(board: BoardState) -> bool:
    """Whether known history reaches the experiment's two-pass endpoint."""
    return board.has_two_pass_history()


def filter_preterminal_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Exclude terminal/past-terminal histories before selecting evaluation states."""
    accepted, rejected = [], []
    for index, record in enumerate(records):
        if ("moves" not in record or "initial_stones" not in record
                or record.get("move_number") != len(record["moves"])):
            raise ValueError("Evaluation positions require complete, consistent state histories")
        board = board_from_example(record)
        if terminal_history(board):
            rejected.append({"source_row": index, "position_id": position_key(record),
                             "game_id": record["game_id"], "move_number": record.get("move_number"),
                             "reason": "history_reaches_two_pass_termination"})
        else:
            accepted.append(record)
    return accepted, rejected


def _estimate(result: Any, player: str) -> dict[str, Any]:
    if result.score_lead is None or not math.isfinite(result.score_lead):
        raise ValueError("Move-quality evaluation requires a finite child scoreLead")
    if not math.isfinite(result.winrate) or not 0 <= result.winrate <= 1:
        raise ValueError("Move-quality evaluation requires a finite winrate in [0,1]")
    return {
        "winrate": result.winrate if player == "B" else 1 - result.winrate,
        "score_lead": result.score_lead * (1 if player == "B" else -1),
        "perspective": player, "visits": result.raw.get("rootInfo", {}).get("visits"),
        "is_estimate": True,
    }


def score_move_predictions(
    records: Sequence[Mapping[str, Any]], predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    engine: Any, *, max_visits: int = 32, batch_size: int = 8, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Analyze fresh root references and every unique candidate child equally.

    Models must be run before calling this function. It never substitutes an
    engine move for a model prediction. Invalid predictions have no quality
    estimate and are included in legality metrics, not silently dropped.
    """
    if not records or not predictions or max_visits < 1 or batch_size < 1:
        raise ValueError("Need positions, predictions and positive analysis budgets")
    if {"uniform", "reference"} & set(predictions):
        raise ValueError("uniform and reference are reserved evaluation names")
    if any(len(values) != len(records) for values in predictions.values()):
        raise ValueError("Each model needs exactly one prediction per position")
    boards = [board_from_example(record) for record in records]
    if any(terminal_history(board) for board in boards):
        raise ValueError("Move-quality positions must precede two-pass termination")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        legality_superko(record["rules"])
        if not math.isfinite(float(record["komi"])):
            raise ValueError("Komi must be finite")
        groups[json.dumps([record["rules"], record["komi"]], sort_keys=True)].append(index)
    roots: dict[int, Any] = {}
    analysis_count = 0
    for indices in groups.values():
        for offset in range(0, len(indices), batch_size):
            chunk = indices[offset:offset + batch_size]
            record = records[chunk[0]]
            results = engine.analyze_positions([boards[index] for index in chunk], max_visits=max_visits,
                                               rules=record["rules"], komi=float(record["komi"]))
            roots.update(zip(chunk, results, strict=True))
            analysis_count += len(chunk)
    rows = []
    children: dict[tuple[int, str], Any] = {}
    for index, (record, board) in enumerate(zip(records, boards, strict=True)):
        legal = board.legal_moves(superko=legality_superko(record["rules"]))
        reference = (roots[index].best_move or "").strip().upper()
        if reference not in legal:
            raise ValueError("Reference root search did not return a legal move")
        rng = random.Random(int(stable_digest([seed, position_key(record), "uniform"]), 16))
        agents = {name: dict(values[index]) for name, values in predictions.items()}
        agents.update(reference={"move": reference}, uniform={"move": rng.choice(legal)})
        for prediction in agents.values():
            move = str(prediction.get("move", "")).strip().upper()
            prediction.update(move=move, legal=move in legal, passed=move == "PASS",
                              reference_match=move == reference)
            if move in legal and (index, move) not in children:
                child = board.clone()
                child.apply_move(board.next_player(), move, strict=True)
                children[index, move] = child
        rows.append({
            "position_id": position_key(record), "game_id": record["game_id"],
            "position": {**state_record_from_board(board), "game_id": record["game_id"],
                         "rules": record["rules"], "komi": record["komi"], "split": record.get("split")},
            "reference_move": reference, "root_estimate": _estimate(roots[index], board.next_player()),
            "agents": agents,
        })
    evaluations: dict[tuple[int, str], Any] = {}
    for indices in groups.values():
        selected = set(indices)
        keys = [key for key in children if key[0] in selected]
        record = records[indices[0]]
        for offset in range(0, len(keys), batch_size):
            chunk = keys[offset:offset + batch_size]
            results = engine.analyze_positions([children[key] for key in chunk], max_visits=max_visits,
                                               rules=record["rules"], komi=float(record["komi"]))
            evaluations.update(zip(chunk, results, strict=True))
            analysis_count += len(chunk)
    for index, row in enumerate(rows):
        player = boards[index].next_player()
        reference = _estimate(evaluations[index, row["reference_move"]], player)
        uniform_move = row["agents"]["uniform"]["move"]
        uniform = _estimate(evaluations[index, uniform_move], player)
        for prediction in row["agents"].values():
            prediction["child_estimate"] = None
            if prediction["legal"]:
                child = _estimate(evaluations[index, prediction["move"]], player)
                prediction.update(
                    child_estimate=child,
                    score_difference_vs_reference=child["score_lead"] - reference["score_lead"],
                    winrate_difference_vs_reference=child["winrate"] - reference["winrate"],
                    score_difference_vs_uniform=child["score_lead"] - uniform["score_lead"],
                    winrate_difference_vs_uniform=child["winrate"] - uniform["winrate"],
                )
    return rows, {"root_queries": len(records), "unique_child_queries": len(children),
                  "total_queries": analysis_count, "max_visits_per_query": max_visits}


def game_cluster_interval(
    observations: Sequence[tuple[str, float]], *, samples: int = 2000, seed: int = 42,
) -> dict[str, Any]:
    """Equal-game mean and percentile 95% CI, resampling complete game clusters.

    Each game's positions are first averaged; games then receive equal weight.
    This captures sampling uncertainty across source games, not engine error or
    variability across training seeds. One game cannot estimate that uncertainty.
    """
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for game_id, value in observations:
        if not game_id or not math.isfinite(value):
            raise ValueError("Bootstrap observations need game IDs and finite values")
        grouped[game_id].append(value)
    means = [sum(grouped[key]) / len(grouped[key]) for key in sorted(grouped)]
    result = {"mean": sum(means) / len(means) if means else None,
              "ci95": None, "games": len(means), "positions": len(observations),
              "weighting": "equal_game_mean", "bootstrap_samples": samples}
    if len(means) >= 2:
        rng = random.Random(seed)
        draws = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(samples))

        def quantile(q: float) -> float:
            location = (samples - 1) * q
            lo, hi = math.floor(location), math.ceil(location)
            return draws[lo] + (draws[hi] - draws[lo]) * (location - lo)

        result["ci95"] = [quantile(.025), quantile(.975)]
    return result


def summarize_move_quality(rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 2000,
                           seed: int = 42) -> dict[str, Any]:
    if not rows:
        raise ValueError("Need at least one evaluated position")
    metrics = ("legal", "passed", "reference_match", "score_difference_vs_reference",
               "winrate_difference_vs_reference", "score_difference_vs_uniform", "winrate_difference_vs_uniform")
    summaries = {}
    for name in rows[0]["agents"]:
        summaries[name] = {
            metric: game_cluster_interval([
                (row["game_id"], float(row["agents"][name][metric]))
                for row in rows if metric in row["agents"][name]
            ], samples=bootstrap_samples, seed=seed)
            for metric in metrics
        }
    return summaries
