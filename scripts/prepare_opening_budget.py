"""Freeze an explicit opening/state-coverage quota without model/engine queries."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256, read_records, state_identity, validate_splits
from go_rl.indexed_jsonl import IndexedJsonlRecords, write_record_cache
from go_rl.rules import normalize_rules
from go_rl.teacher_data import TARGET_RULES, select_game_balanced, select_phase_balanced, validate_history_for_rules


def validate_exclusion_objectives(records):
    """Hash exclusions only under the same objective used for candidate states."""
    for row in records:
        if row.get("split") not in {"validation", "test"}:
            raise ValueError("Exclusions must be pinned to validation or test")
        if normalize_rules(row["rules"]) != normalize_rules(TARGET_RULES) or float(row["komi"]) != 7.5:
            raise ValueError("Exclusions must use the target rules and komi 7.5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/sft_19x19_small.jsonl")
    parser.add_argument("--background", default="data/teacher_19x19_train_v2_source.jsonl")
    parser.add_argument("--exclude-dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error("Frozen output already exists")
    heldout = [row for path in args.exclude_dataset for row in read_records(path)]
    validate_exclusion_objectives(heldout)
    heldout_ids = {row["game_id"] for row in heldout}
    forbidden = {state_identity(row) for row in heldout}
    source_games, seen, rejected = set(), set(), Counter()
    empty, opening, background = [], [], []
    for source, phase in ((args.dataset, "opening"), (args.background, "background")):
        for index, row in enumerate(IndexedJsonlRecords(source)):
            if row.get("split") != "train":
                continue
            source_games.add(row["game_id"])
            number = row.get("move_number", -1)
            if (phase == "opening" and not 0 <= number <= 20) or (phase == "background" and number <= 20):
                continue
            if row["game_id"] in heldout_ids:
                raise ValueError("Source training game overlaps held-out games")
            key = (row["game_id"], number)
            if key in seen:
                rejected["duplicate_game_move"] += 1
                continue
            try:
                if "initial_stones" not in row or "moves" not in row or number != len(row["moves"]):
                    raise ValueError("Incomplete history")
                board = board_from_example(row)
                if board.size != 19:
                    raise ValueError("Not 19x19")
                if any(first[1] is None and second[1] is None for first, second in zip(board.moves, board.moves[1:])):
                    raise ValueError("Terminal history")
                validate_history_for_rules(board, TARGET_RULES)
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            if state_identity({**row, "rules": TARGET_RULES, "komi": 7.5}) in forbidden:
                rejected["heldout_visible_state"] += 1
                continue
            seen.add(key)
            if phase == "background":
                background.append(row)
            elif number == 0:
                if not row["initial_stones"]:
                    empty.append(row)
                else:
                    rejected["move_zero_setup_not_empty"] += 1
            else:
                opening.append(row)
    starts = select_game_balanced(empty, 8, args.seed, per_game=1) if empty else []
    early = select_game_balanced(opening, 64 - len(starts), args.seed, per_game=64)
    later = select_phase_balanced(background, 448, args.seed, per_game=16)
    selected = [*starts, *early, *later]
    if len(early) != 64 - len(starts) or len(later) != 448 or len(selected) != 512:
        raise ValueError("Insufficient eligible real source positions for the fixed coverage quota")
    if {row["game_id"] for row in selected} != source_games:
        raise ValueError("Coverage quota did not retain every source training game")
    target_rows = [{**row, "rules": TARGET_RULES, "komi": 7.5} for row in selected]
    validate_splits(target_rows, heldout)
    write_record_cache(selected, output)
    manifest = {
        "purpose": "opening_and_full_game_state_coverage_teacher_budget", "split": "train",
        "source_datasets": {path: file_sha256(path) for path in (args.dataset, args.background)},
        "excluded_datasets": {path: file_sha256(path) for path in args.exclude_dataset},
        "selected": len(selected), "games": len(source_games), "selection_seed": args.seed,
        "target_rules": TARGET_RULES, "target_komi": 7.5,
        "quota": {"empty_move_zero_requested": 8, "empty_move_zero_selected": len(starts),
                  "moves_1_20_selected": len(early), "moves_over_20_selected": len(later)},
        "eligible": {"empty_move_zero": len(empty), "moves_1_20": len(opening), "moves_over_20": len(background)},
        "selected_move_bins": dict(Counter("0-5" if row["move_number"] <= 5 else "6-20" if row["move_number"] <= 20
                                           else "21+" for row in selected)),
        "selected_positions": [{"game_id": row["game_id"], "move_number": row["move_number"]} for row in selected],
        "selected_game_counts": dict(Counter(row["game_id"] for row in selected)),
        "rejections_before_sampling": dict(rejected),
        "deduplication": "exact game_id plus move_number only; repeated openings across real training games retained",
        "selection_used_teacher_outputs": False, "engine_or_model_queries": False,
        "source_fields_preserved": True, "output_sha256": file_sha256(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("selected", "games", "quota", "eligible", "selected_move_bins", "output_sha256")}))


if __name__ == "__main__":
    main()
