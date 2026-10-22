"""Export only a named learner's turns from match SGFs for teacher reanalysis.

Example:
  uv run python scripts/export_learner_sgf.py --sgf-dir outputs/matches/sgf \
    --learner-name qwen-text --checkpoint outputs/text_distill/best \
    --exclude-dataset data/teacher_19x19_val_v1.jsonl \
    --output data/text_learner_refresh_source.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from go_rl.experiment_utils import file_sha256, read_records
from go_rl.learner_data import deduplicate_training_records
from go_rl.learner_sgf import sgf_training_records
from go_rl.teacher_data import select_game_balanced
from go_rl.training_checkpoint import model_fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf-dir", action="append", required=True)
    parser.add_argument("--learner-name", required=True, help="Exact PB/PW player name")
    parser.add_argument("--checkpoint", help="Optional immutable learner snapshot used for these matches")
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--max-positions", type=int, help="Optional total budget after deduplication and exclusions")
    parser.add_argument("--per-game", type=int, help="Optional per-game budget after deduplication and exclusions")
    parser.add_argument("--seed", type=int, default=3407, help="Seed for optional game-balanced sampling")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in (args.max_positions, args.per_game)):
        parser.error("Sampling budgets must be positive")
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error("Output exists; choose a fresh path to preserve provenance")
    paths = sorted({path.resolve() for directory in args.sgf_dir
                    for path in Path(directory).rglob("*.sgf")})
    if not paths:
        parser.error("No SGF files found")
    checkpoint = {"path": args.checkpoint, "fingerprint_observed_at_export": True,
                  "available_at_export": bool(args.checkpoint)}
    if args.checkpoint:
        checkpoint["files_sha256"] = model_fingerprints(Path(args.checkpoint))
    records, sources = [], []
    for path in paths:
        data = path.read_bytes()
        selected = sgf_training_records(data, learner_name=args.learner_name,
                                        source_file=str(path), checkpoint=checkpoint)
        records.extend(selected)
        sources.append({"path": str(path), "sha256": file_sha256(path),
                        "learner_positions": len(selected), "game_id": selected[0]["game_id"],
                        "black_name": selected[0]["learner"]["black_name"],
                        "white_name": selected[0]["learner"]["white_name"],
                        "sgf_result": selected[0]["learner"]["sgf_result"]})
    exclusions = [row for path in args.exclude_dataset for row in read_records(path)]
    eligible, counts = deduplicate_training_records(records, excluded_records=exclusions)
    if not eligible:
        parser.error("No distinct learner positions remain after exclusion")
    bounded = args.max_positions is not None or args.per_game is not None
    selected = (select_game_balanced(eligible, args.max_positions or len(eligible), args.seed,
                                     per_game=args.per_game or len(eligible)) if bounded else eligible)
    counts.update(eligible_positions_after_dedup=len(eligible),
                  eligible_games_after_dedup=len({row["game_id"] for row in eligible}),
                  budget_excluded_positions=len(eligible) - len(selected),
                  selected_positions=len(selected), exported_positions=len(selected))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in selected))
    temporary.replace(output)
    manifest = {
        "purpose": "learner_turns_from_gtp_matches_for_teacher_reanalysis",
        "schema_version": 2, "split": "train", "learner_name": args.learner_name,
        **counts, "games": len({row["game_id"] for row in selected}), "sources": sources,
        "checkpoint": checkpoint,
        "excluded_datasets": {path: file_sha256(path) for path in args.exclude_dataset},
        "sampled_actions_are_expert_labels": False,
        "deduplication": "visible board up to D4, side-to-play, rules, komi; first history retained",
        "sampling": {"method": "round_robin_games_with_seeded_shuffle" if bounded else "all_eligible_source_order",
                     "max_positions": args.max_positions, "per_game": args.per_game, "seed": args.seed,
                     "applied_after": "deduplication_and_heldout_state_exclusion",
                     "uses_move_quality_or_outcomes": False},
        "dataset_sha256": file_sha256(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), **counts, "dataset_sha256": manifest["dataset_sha256"]}))


if __name__ == "__main__":
    main()
