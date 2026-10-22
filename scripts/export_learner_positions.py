"""Export learner-visited states for teacher reanalysis, without model inference.

Example:
  uv run python scripts/export_learner_positions.py --run-dir outputs/policy_rl \
    --exclude-dataset data/teacher_19x19_val_v1.jsonl \
    --output data/learner_refresh_source.jsonl

Then pass the exported source to build_teacher_dataset.py. It has no expert
response targets until annotation; the learner's own actions are provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from go_rl.experiment_utils import file_sha256, read_records, stable_digest
from go_rl.learner_data import deduplicate_training_records, episode_training_records


def checkpoint_provenance(path: str | None) -> dict:
    """Observe the immutable starting checkpoint, never invent a missing hash."""
    result = {"path": path, "fingerprint_observed_at_export": True}
    if not path or not Path(path).is_dir():
        return {**result, "available_at_export": False}
    directory = Path(path)
    files = sorted({*directory.rglob("*.safetensors"), *directory.glob("go_policy.json")})
    hashes = {str(item.relative_to(directory)): file_sha256(item) for item in files}
    return {**result, "available_at_export": True, "files_sha256": hashes,
            "content_sha256": stable_digest(hashes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--exclude-dataset", action="append", default=[],
                        help="Remove matching validation states using no held-out labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error("Output exists; choose a fresh path to preserve provenance")
    records, sources = [], []
    for directory in sorted(set(args.run_dir)):
        run_dir = Path(directory)
        run_path, episodes_path = run_dir / "rl_run.json", run_dir / "episodes.jsonl"
        run = json.loads(run_path.read_text())
        episodes = read_records(episodes_path)
        hashes = {"rl_run_sha256": file_sha256(run_path),
                  "episodes_sha256": file_sha256(episodes_path)}
        run_id = stable_digest(hashes)
        checkpoint = run.get("starting_checkpoint") or checkpoint_provenance(run.get("model"))
        for index, episode in enumerate(episodes):
            records.extend(episode_training_records(
                episode, run, source_run_id=run_id, episode_row=index, checkpoint=checkpoint))
        sources.append({"run_dir": str(run_dir), "source_run_id": run_id, **hashes,
                        "episodes": len(episodes), "starting_checkpoint": checkpoint})
    exclusions = [row for path in args.exclude_dataset for row in read_records(path)]
    records, counts = deduplicate_training_records(records, excluded_records=exclusions)
    if not records:
        parser.error("No distinct training positions remain after exclusion")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in records))
    temporary.replace(output)
    manifest = {
        "purpose": "learner_visited_training_source_for_teacher_reanalysis",
        "schema_version": 1, "split": "train", **counts,
        "games": len({row["game_id"] for row in records}), "sources": sources,
        "excluded_datasets": {path: file_sha256(path) for path in args.exclude_dataset},
        "deduplication": "visible board up to D4, side-to-play, rules, komi; first history retained",
        "sampled_actions_are_expert_labels": False, "teacher_or_model_evaluated": False,
        "dataset_sha256": file_sha256(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), **counts, "dataset_sha256": manifest["dataset_sha256"]}))


if __name__ == "__main__":
    main()
