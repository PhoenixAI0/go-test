"""KataGo targets for an explicit finite-action learning problem."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .board import BoardState
from .katago import KataGoResult
from .model_interface import legal_action_mask, move_to_action
from .rules import legality_superko, normalize_rules

TARGET_RULES = "koPOSITIONALscoreAREAtaxNONEsui0"


def validate_history_for_rules(board: BoardState, rules: str | Mapping[str, object]) -> None:
    """Reject incompatible replay before reanalyzing under a different objective.

    BoardState already checks captures, suicide, and simple ko when constructed
    by board_from_example. Positional superko additionally checks every non-pass
    position against ALL known earlier positions. Passes are exempt.
    """
    if not legality_superko(rules):
        return
    history = board.position_history()
    offset = len(history) - len(board.moves)
    known = {tuple(tuple(row) for row in state) for state in history[:offset]}
    for (_, coords), state in zip(board.moves, history[offset:], strict=True):
        key = tuple(tuple(row) for row in state)
        if coords is not None and key in known:
            raise ValueError("Source history violates target positional superko")
        known.add(key)


def make_teacher_record(
    source: Mapping[str, Any], board: BoardState, analysis: KataGoResult, *,
    rules: str = TARGET_RULES, komi: float = 7.5, prior_mix: float = 1.0,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Mix searched visit targets and a dense prior, preserving target provenance.

    A small-budget engine is an approximate teacher, never an exact win verifier.
    Raw engine winrates/score leads use BLACK perspective; teacher_value is the
    current player's signed [-1,1] value. Source-game results are not relabeled.
    """
    if not math.isfinite(prior_mix) or not 0 <= prior_mix <= 1:
        raise ValueError("prior_mix must be in [0,1]")
    validate_history_for_rules(board, rules)
    legal = legal_action_mask(board, superko=legality_superko(rules))
    count = len(legal)
    raw_policy = analysis.raw.get("policy")
    if not isinstance(raw_policy, list) or len(raw_policy) != count:
        raise ValueError("Teacher must return a dense policy with one entry per action")
    prior = [float(value) for value in raw_policy]
    if not all(math.isfinite(value) for value in prior):
        raise ValueError("Teacher policy contains nonfinite values")
    # KataGo marks illegal actions as negative. A disagreement is a verifier bug,
    # not something to silently mask away in a learning experiment.
    if any((value >= 0) != allowed for value, allowed in zip(prior, legal, strict=True)):
        raise ValueError("Local and KataGo legal-action masks disagree")
    prior = [max(0.0, value) for value in prior]
    mass = sum(prior)
    if mass <= 0:
        raise ValueError("Teacher policy has no probability mass")
    prior = [value / mass for value in prior]
    move_infos = analysis.raw.get("moveInfos", [])
    visits = [0.0] * count
    for info in move_infos:
        action = move_to_action(info["move"], board.size)
        visit_count = float(info.get("visits", 0))
        if not legal[action] or not math.isfinite(visit_count) or visit_count < 0:
            raise ValueError("Invalid teacher move/visit count")
        visits[action] += visit_count
    visit_mass = sum(visits)
    if prior_mix < 1 and visit_mass <= 0:
        raise ValueError("Search targets require positive visit counts")
    target = [prior_mix * prior[i] + (1 - prior_mix) * visits[i] / visit_mass
              if visit_mass > 0 else prior[i] for i in range(count)]
    if not math.isfinite(analysis.winrate) or not 0 <= analysis.winrate <= 1:
        raise ValueError("Teacher winrate must lie in [0,1]")
    best = analysis.best_move
    if not best or not legal[move_to_action(best, board.size)]:
        raise ValueError("Teacher returned no legal best move")
    result = dict(source)
    for key in ("rules", "komi", "response", "result", "opponent_response"):
        if key in result:
            result[f"source_{key}"] = result.pop(key)
    result.update(
        rules=rules, komi=komi, response=best,
        teacher_labeled=True,
        policy_target=target, legal_mask=legal, teacher_best_move=best,
        teacher_value=(2 * analysis.winrate - 1) * (1 if board.next_player() == "B" else -1),
        teacher_move_infos=move_infos,
        teacher={**dict(provenance or {}), "schema_version": 2,
                 "perspective": "black", "value_perspective": "side_to_move",
                 "target_type": "prior" if prior_mix == 1 else "visits" if prior_mix == 0 else "mixed",
                 "prior_mix": prior_mix, "normalized_rules": normalize_rules(rules),
                 "outcome_is_estimate": True},
    )
    return result


def select_game_balanced(records: Sequence[Mapping[str, Any]], count: int, seed: int,
                         per_game: int = 8) -> list[dict[str, Any]]:
    indices = select_game_balanced_indices(records, count, seed, per_game)
    return [dict(records[index]) for index in indices]


def select_game_balanced_indices(
    records: Sequence[Mapping[str, Any]], count: int, seed: int, per_game: int = 8,
    *, predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[int]:
    """Keep compact row indices while preserving the original shuffle exactly."""
    from array import array
    import random
    from collections import defaultdict

    if count < 1 or per_game < 1:
        raise ValueError("count and per_game must be positive")
    grouped = defaultdict(lambda: array("Q"))
    for index, record in enumerate(records):
        if predicate is not None and not predicate(record):
            continue
        if not record.get("game_id"):
            raise ValueError("Research datasets require game_id for independent sampling")
        grouped[record["game_id"]].append(index)
    rng = random.Random(seed)
    games = sorted(grouped)
    rng.shuffle(games)
    for game in games:
        rng.shuffle(grouped[game])
    selected = []
    for index in range(per_game):
        for game in games:
            if index < len(grouped[game]):
                selected.append(grouped[game][index])
                if len(selected) == count:
                    return selected
    return selected


def move_phase(move_number: int) -> str:
    """Fixed 19x19 phase strata; these are sampling bins, not Go judgments."""
    if move_number < 0:
        raise ValueError("move_number must be nonnegative")
    return ("0-60" if move_number <= 60 else "61-180" if move_number <= 180
            else "181-280" if move_number <= 280 else "281+")


def select_phase_balanced(
    records: Sequence[Mapping[str, Any]], count: int, seed: int, per_game: int = 8,
) -> list[dict[str, Any]]:
    """Round-robin games and their available move phases without using labels.

    Each game's phase order receives a deterministic offset so a budget of one
    position per game does not always sample openings. Exhausted/missing phases
    are skipped. Positions are shuffled by hashes within phases; duplicate rows
    and input ordering cannot change selection. Split before calling this.
    """
    indices = select_phase_balanced_indices(records, count, seed, per_game)
    return [dict(records[index]) for index in indices]


def select_phase_balanced_indices(
    records: Sequence[Mapping[str, Any]], count: int, seed: int, per_game: int = 8,
    *, predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[int]:
    """Exact phase-balanced selection with bounded per-game hash reservoirs.

    A game can supply at most ``per_game`` rows, so no phase can need more than
    that many lowest-hash candidates. Only their row indices and keys survive
    the streaming pass. A duplicate retained state updates its index to the last
    row, matching the original dictionary semantics without retaining row data.
    """
    from collections import defaultdict
    import heapq
    from .experiment_utils import stable_digest

    if count < 1 or per_game < 1:
        raise ValueError("count and per_game must be positive")
    phases = ["0-60", "61-180", "181-280", "281+"]
    limit = min(count, per_game)
    grouped = defaultdict(lambda: defaultdict(lambda: ({}, [])))
    for index, record in enumerate(records):
        if predicate is not None and not predicate(record):
            continue
        game = record.get("game_id")
        if not isinstance(game, str) or not game:
            raise ValueError("Research datasets require game_id for independent sampling")
        number = record.get("move_number")
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError("Phase sampling requires integer move_number metadata")
        phase = move_phase(number)
        # Selection keys deliberately omit response/teacher targets. Curriculum
        # allocation must not depend on whether a teacher likes a move or passes.
        key = stable_digest([game, number, record.get("prompt"), record.get("moves"),
                             record.get("initial_stones"), record.get("rules"), record.get("komi")])
        bucket, heap = grouped[game][phase]
        if key in bucket:
            bucket[key] = index
            continue
        priority = stable_digest([seed, "position", key])
        rank = (-int(priority, 16), -int(key, 16), key)
        if len(bucket) < limit:
            bucket[key] = index
            heapq.heappush(heap, rank)
        elif rank > heap[0]:
            removed = heapq.heapreplace(heap, rank)
            del bucket[removed[2]]
            bucket[key] = index
    games = sorted(grouped, key=lambda game: (stable_digest([seed, "game", game]), game))
    ordered = {}
    for game in games:
        queues = {phase: [bucket[key] for key in sorted(
            bucket, key=lambda key: (stable_digest([seed, "position", key]), key))]
            for phase, (bucket, _) in grouped[game].items()}
        offset = int(stable_digest([seed, "phase-offset", game])[:16], 16) % len(phases)
        phase_order = phases[offset:] + phases[:offset]
        next_index = defaultdict(int)
        positions = []
        while len(positions) < per_game:
            added = False
            for phase in phase_order:
                if next_index[phase] < len(queues.get(phase, [])):
                    positions.append(queues[phase][next_index[phase]])
                    next_index[phase] += 1
                    added = True
                    if len(positions) == per_game:
                        break
            if not added:
                break
        ordered[game] = positions
    selected = []
    for index in range(per_game):
        for game in games:
            if index < len(ordered[game]):
                selected.append(ordered[game][index])
                if len(selected) == count:
                    return selected
    return selected
