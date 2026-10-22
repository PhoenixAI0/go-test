"""Freeze unused archive games for final testing without engine/model analysis.

Selection follows archive order, excludes imported content hashes and canonical
game IDs, and uniformly samples positions using a fixed seed. Do not use these
records for model selection or training; annotate them only after model choice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from go_rl.data_prep import run_go_parser
from go_rl.experiment_utils import file_sha256, read_records
from go_rl.sgf_download import (
    ARCHIVE_SHA256, ARCHIVE_URL, MAX_SGF_BYTES, fetch_archive, sgf_board_size,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-dataset", action="append", required=True)
    parser.add_argument("--exclude-source-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--positions-per-game", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if min(args.games, args.positions_per_game) < 1:
        parser.error("Game and position counts must be positive")
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error("Output already exists; frozen test data must not be overwritten")
    excluded_ids = {row["game_id"] for path in args.exclude_dataset for row in read_records(path)}
    excluded_files = {name for path in args.exclude_source_manifest
                      for name in json.loads(Path(path).read_text())["files"]}
    archive = fetch_archive(Path(args.cache_dir))
    source_hashes = []
    with tempfile.TemporaryDirectory(prefix="go-final-test-") as temporary:
        directory = Path(temporary)
        sgf_dir = directory / "sgf"
        sgf_dir.mkdir()
        with tarfile.open(archive, "r|bz2") as bundle:
            for member in bundle:
                if not member.isfile() or not member.name.lower().endswith(".sgf") or member.size > MAX_SGF_BYTES:
                    continue
                with bundle.extractfile(member) as handle:
                    data = handle.read(MAX_SGF_BYTES + 1)
                filename = hashlib.sha256(data).hexdigest() + ".sgf"
                if filename in excluded_files or filename in source_hashes:
                    continue
                try:
                    if sgf_board_size(data) != 19:
                        continue
                except (ValueError, KeyError, UnicodeError):
                    continue
                (sgf_dir / filename).write_bytes(data)
                source_hashes.append(filename)
                # Extra candidates permit malformed/duplicate games to be skipped
                # without using their move quality or engine labels for selection.
                if len(source_hashes) >= args.games * 2:
                    break
        raw_path = directory / "raw.jsonl"
        run_go_parser(str(sgf_dir), str(raw_path), len(source_hashes) * args.positions_per_game,
                      num_workers=4, seed=args.seed, positions_per_game=args.positions_per_game)
        records_by_source = defaultdict(list)
        for row in read_records(raw_path):
            records_by_source[Path(row["source_file"]).name].append(row)
        selected, selected_files, selected_ids = [], [], set()
        for filename in source_hashes:
            records = records_by_source[filename]
            if not records or records[0]["game_id"] in excluded_ids | selected_ids:
                continue
            selected_ids.add(records[0]["game_id"])
            selected_files.append(filename)
            for row in records:
                row["source_file"] = f"{ARCHIVE_URL}#{filename}"
                row["split"] = "test"
                selected.append(row)
            if len(selected_ids) == args.games:
                break
        if len(selected_ids) != args.games:
            raise RuntimeError("Not enough unused canonical games; no final-test files were published")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in selected))
    parser_dir = Path(__file__).resolve().parents[1] / "go_parser"
    parser_hash = hashlib.sha256()
    for source in sorted(parser_dir.glob("*.go")):
        parser_hash.update(source.name.encode())
        parser_hash.update(source.read_bytes())
    manifest = {
        "purpose": "untouched_final_test", "split": "test", "board_size": 19,
        "source_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
        "selection": "first unused canonical games in archive order; no engine/model analysis",
        "games": len(selected_ids), "positions": len(selected),
        "positions_per_game": args.positions_per_game, "seed": args.seed,
        "source_files": selected_files, "game_ids": sorted(selected_ids),
        "excluded_datasets": {path: file_sha256(path) for path in args.exclude_dataset},
        "excluded_source_manifests": {path: file_sha256(path) for path in args.exclude_source_manifest},
        "parser_source_sha256": parser_hash.hexdigest(), "dataset_sha256": file_sha256(output),
        "engine_or_model_evaluated": False,
        "release_condition": "Freeze architecture, hyperparameters and checkpoint choice before annotation/evaluation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path),
                      "games": len(selected_ids), "positions": len(selected),
                      "dataset_sha256": manifest["dataset_sha256"]}))


if __name__ == "__main__":
    main()
