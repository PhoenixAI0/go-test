"""Freeze existing test histories under an explicit objective, without analysis.

No model or engine is loaded. Selection depends only on complete state validity
and the first two-pass boundary; source moves and outcomes remain provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path
from typing import Any

from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256
from go_rl.indexed_jsonl import IndexedJsonlRecords, write_record_cache
from go_rl.rules import legality_superko, normalize_rules
from go_rl.teacher_data import TARGET_RULES, validate_history_for_rules


def target_test_records(
    records: Iterable[Mapping[str, Any]], *, rules: str, komi: float,
    board_size: int, source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain source order, rejecting state incompatibilities without replacing rows."""
    legality_superko(rules)
    if not math.isfinite(komi):
        raise ValueError("Target komi must be finite")
    accepted, rejected = [], []
    for index, source in enumerate(records):
        if source.get("split") != "test" or not isinstance(source.get("game_id"), str) or not source["game_id"]:
            raise ValueError("Frozen evaluation sources require explicit test split and game_id")
        if any(key.startswith("teacher") or key in {"policy_target", "legal_mask", "evaluation_target"}
               for key in source):
            raise ValueError("Use the original frozen source, without teacher labels or prior relabeling")
        reason = "invalid_state_history"
        try:
            if source.get("board_size") != board_size:
                raise ValueError("Board size differs from the frozen objective")
            if "moves" not in source or "initial_stones" not in source:
                raise ValueError("Evaluation records require complete setup and move history")
            if (type(source.get("move_number")) is not int
                    or source["move_number"] != len(source["moves"])):
                raise ValueError("move_number differs from complete history")
            board = board_from_example(source)
            if source.get("to_play") != board.next_player():
                raise ValueError("to_play differs from replayed state")
            if any(first[1] is None and second[1] is None
                   for first, second in zip(board.moves, board.moves[1:])):
                reason = "terminal_history"
                raise ValueError("History reaches or continues after two-pass termination")
            reason = "incompatible_target_rules"
            validate_history_for_rules(board, rules)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"source_row": index, "game_id": source["game_id"],
                             "move_number": source.get("move_number"),
                             "category": reason, "reason": str(exc)})
            continue
        record = dict(source)
        for key in ("rules", "komi", "response", "result", "opponent_response"):
            if key in record:
                if f"source_{key}" in record:
                    raise ValueError(f"Source already has conflicting provenance field source_{key}")
                record[f"source_{key}"] = record.pop(key)
        record.update(rules=rules, komi=komi, teacher_labeled=False,
                      evaluation_target={"source_dataset_sha256": source_sha256,
                                         "source_row": index,
                                         "objective_relabeled": True,
                                         "engine_or_model_evaluated": False})
        accepted.append(record)
    return accepted, rejected


def freeze_target(source_path: Path, source_manifest_path: Path, output: Path, *,
                  rules: str = TARGET_RULES, komi: float = 7.5) -> dict[str, Any]:
    manifest_path = output.with_suffix(".manifest.json")
    if any(path.exists() for path in (output, manifest_path, output.with_suffix(output.suffix + ".tmp"),
                                      manifest_path.with_suffix(manifest_path.suffix + ".tmp"))):
        raise ValueError("Frozen output or temporary file exists; choose a fresh path")
    source_manifest_sha = file_sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    source_sha = file_sha256(source_path)
    if source_manifest.get("dataset_sha256") != source_sha:
        raise ValueError("Frozen source hash differs from its manifest")
    if source_manifest.get("split") != "test" or source_manifest.get("engine_or_model_evaluated") is not False:
        raise ValueError("Source manifest must describe untouched test data")
    source = IndexedJsonlRecords(source_path)
    source_games = {row["game_id"] for row in source}
    if len(source) != source_manifest["positions"] or len(source_games) != source_manifest["games"]:
        raise ValueError("Frozen source counts differ from its manifest")
    if sorted(source_games) != sorted(source_manifest["game_ids"]):
        raise ValueError("Frozen source game IDs differ from its manifest")
    rows, rejected = target_test_records(
        source, rules=rules, komi=komi, board_size=source_manifest["board_size"], source_sha256=source_sha)
    if not rows:
        raise ValueError("No compatible preterminal test histories; no output published")
    if file_sha256(source_path) != source_sha or file_sha256(source_manifest_path) != source_manifest_sha:
        raise ValueError("Frozen source changed during preparation")
    published = write_record_cache(rows, output)
    games = Counter(row["game_id"] for row in rows)
    manifest = {
        "purpose": "untouched_final_test_target_objective", "split": "test",
        "board_size": source_manifest["board_size"], "rules": rules,
        "normalized_rules": normalize_rules(rules), "komi": komi,
        "positions": len(published), "games": len(games), "positions_per_game": dict(sorted(games.items())),
        "source_dataset": str(source_path), "source_dataset_sha256": source_sha,
        "source_manifest_path": str(source_manifest_path), "source_manifest_sha256": source_manifest_sha,
        "source_manifest": source_manifest,
        "source_positions": len(source), "source_games": len(source_games),
        "selection": "original source order; state/history validity only; no replacement or resampling",
        "terminal_filter": "exclude any history reaching two consecutive passes",
        "rejection_count": len(rejected), "rejection_counts": dict(Counter(row["category"] for row in rejected)),
        "rejected": rejected, "removed_game_ids": sorted(source_games - games.keys()),
        "retained_source_rows": [row["evaluation_target"]["source_row"] for row in rows],
        "game_ids": sorted(games), "dataset_sha256": file_sha256(output),
        "engine_or_model_evaluated": False, "teacher_labeled": False,
        "source_outcomes_apply_to_target": False,
        "release_condition": source_manifest.get("release_condition"),
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules", default=TARGET_RULES)
    parser.add_argument("--komi", type=float, default=7.5)
    args = parser.parse_args()
    manifest = freeze_target(Path(args.dataset), Path(args.source_manifest), Path(args.output),
                             rules=args.rules, komi=args.komi)
    print(json.dumps({key: manifest[key] for key in (
        "positions", "games", "rejection_count", "rejection_counts", "dataset_sha256",
        "engine_or_model_evaluated")}))


if __name__ == "__main__":
    main()
