"""Freeze a preterminal subset of existing labels without relabeling or resampling."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256
from go_rl.indexed_jsonl import IndexedJsonlRecords, write_record_cache
from go_rl.teacher_data import validate_history_for_rules


def freeze_preterminal_subset(source_path: Path, source_manifest_path: Path, output: Path,
                              *, split: str) -> dict:
    manifest_path = output.with_suffix(".manifest.json")
    if any(path.exists() for path in (output, manifest_path, output.with_suffix(output.suffix + ".tmp"),
                                      manifest_path.with_suffix(manifest_path.suffix + ".tmp"))):
        raise ValueError("Frozen output or temporary file exists; choose a fresh path")
    source_sha, manifest_sha = file_sha256(source_path), file_sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_sha != source_manifest.get("output_sha256", source_manifest.get("dataset_sha256")):
        raise ValueError("Source hash differs from the annotation manifest")
    source = IndexedJsonlRecords(source_path)
    retained, rejected, source_games = [], [], set()
    for index, record in enumerate(source):
        if record.get("split") != split or not record.get("game_id"):
            raise ValueError("Every source row must have the requested pinned split and game_id")
        source_games.add(record["game_id"])
        if "moves" not in record or "initial_stones" not in record or record.get("move_number") != len(record["moves"]):
            raise ValueError("Every source row must have a complete, consistent history")
        board = board_from_example(record)
        if any(first[1] is None and second[1] is None for first, second in zip(board.moves, board.moves[1:])):
            rejected.append({"source_row": index, "game_id": record["game_id"],
                             "move_number": record["move_number"], "category": "terminal_history"})
        else:
            validate_history_for_rules(board, record["rules"])
            retained.append(index)
    if not retained:
        raise ValueError("No preterminal positions remain; no output published")
    if file_sha256(source_path) != source_sha or file_sha256(source_manifest_path) != manifest_sha:
        raise ValueError("Source changed during preparation")
    rows = source.select(retained)
    published = write_record_cache(rows, output)
    games = Counter(row["game_id"] for row in rows)
    manifest = {
        "purpose": "preterminal_subset_of_existing_labels", "split": split,
        "positions": len(published), "games": len(games), "game_ids": sorted(games),
        "source_dataset": str(source_path), "source_dataset_sha256": source_sha,
        "source_manifest_path": str(source_manifest_path), "source_manifest_sha256": manifest_sha,
        "source_manifest": source_manifest, "source_positions": len(source), "source_games": len(source_games),
        "selection": "original source order; exclude histories reaching two passes; no replacement or resampling",
        "labels_and_objective_unchanged": True, "new_engine_or_model_queries": False,
        "retained_source_rows": retained, "rejection_count": len(rejected), "rejected": rejected,
        "removed_game_ids": sorted(source_games - games.keys()), "dataset_sha256": file_sha256(output),
        "comparison_note": "Existing validation corpus with terminal rows removed; not a fresh independent holdout",
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
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    args = parser.parse_args()
    manifest = freeze_preterminal_subset(Path(args.dataset), Path(args.source_manifest), Path(args.output), split=args.split)
    print(json.dumps({key: manifest[key] for key in ("positions", "games", "rejection_count", "dataset_sha256")}))


if __name__ == "__main__":
    main()
