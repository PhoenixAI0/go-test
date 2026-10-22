"""Reanalyze game-separated Go positions with a bounded KataGo teacher budget."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from go_rl.board import BoardState
from go_rl.config import KataGoConfig
from go_rl.data_prep import board_from_example
from go_rl.katago import KataGoAnalysisClient
from go_rl.indexed_jsonl import IndexedJsonlRecords
from go_rl.teacher_data import (
    TARGET_RULES, make_teacher_record, move_phase, select_game_balanced_indices,
    select_phase_balanced_indices, validate_history_for_rules,
)


def teacher_batches(records: Sequence[dict], batch_size: int, rules: str,
                    rejected: list[dict]) -> Iterator[list[tuple[dict, BoardState]]]:
    """Replay only a bounded batch; a board caches every historical matrix.

    Keeping all replayed boards is substantially larger than keeping their
    serialized records. Full-game targets also exclude positions at or beyond
    the first two-pass boundary, including source encore/cleanup continuations.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunk = []
    expected_size = None
    for record in records:
        try:
            if "moves" not in record or "initial_stones" not in record:
                raise ValueError("Teacher records require complete initial stones and move history")
            if record.get("move_number") != len(record["moves"]):
                raise ValueError("move_number differs from complete history")
            board = board_from_example(record)
            if any(first[1] is None and second[1] is None
                   for first, second in zip(board.moves, board.moves[1:])):
                raise ValueError("History reaches or continues after two-pass termination")
            validate_history_for_rules(board, rules)
        except ValueError as exc:
            rejected.append({"game_id": record.get("game_id"),
                             "move_number": record.get("move_number"), "reason": str(exc)})
            continue
        if expected_size is not None and board.size != expected_size:
            raise ValueError("Use a single board size per teacher dataset")
        expected_size = board.size
        chunk.append((record, board))
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "validation", "val", "test", "all"], default="train")
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--positions-per-game", type=int, default=8)
    parser.add_argument("--min-move", type=int, default=0,
                        help="Include actual opening states; excluding them can hide from-empty failures")
    parser.add_argument("--max-move", type=int, default=None,
                        help="Optional upper move cutoff; default includes endgame/passing positions")
    parser.add_argument("--sampling", choices=["game", "game-phase"], default="game-phase",
                        help="Game-balanced sampling, optionally cycling move phases within each game")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-visits", type=int, default=8)
    parser.add_argument("--prior-mix", type=float, default=1.0,
                        help="1=dense network prior, 0=searched visits; intermediate values mix them")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--relabel-rules", default=TARGET_RULES,
                        help="Explicit target rules; source metadata and source outcomes are retained separately")
    parser.add_argument("--relabel-komi", type=float, default=7.5)
    parser.add_argument("--katago-bin", default=os.environ.get("KATAGO_BIN", "katago"))
    parser.add_argument("--katago-model", default=os.environ.get("KATAGO_MODEL"), required=not os.environ.get("KATAGO_MODEL"))
    parser.add_argument("--katago-config", required=True)
    args = parser.parse_args()
    if args.split == "val":
        args.split = "validation"
    if args.batch_size < 1 or args.max_visits < 1:
        parser.error("batch-size and max-visits must be positive")
    if args.min_move < 0 or (args.max_move is not None and args.max_move < args.min_move):
        parser.error("Move window must be nonnegative and ordered")
    destination = Path(args.output)
    if destination.exists():
        parser.error("Output exists; choose a new path to preserve experiment provenance")
    source_records = IndexedJsonlRecords(args.dataset)

    def eligible(record):
        return ((args.split == "all" or record.get("split") == args.split)
                and args.min_move <= record.get("move_number", -1)
                and (args.max_move is None or record.get("move_number", -1) <= args.max_move))

    sampler = select_phase_balanced_indices if args.sampling == "game-phase" else select_game_balanced_indices
    indices = sampler(source_records, args.max_positions, args.seed, args.positions_per_game,
                      predicate=eligible)
    records = source_records.select(indices)
    if not records:
        parser.error("No eligible positions; input needs move_number and split metadata")
    rejected = []
    batches = teacher_batches(records, args.batch_size, args.relabel_rules, rejected)
    first_chunk = next(batches, None)
    if not first_chunk:
        parser.error("Every selected history is incompatible with target rules")
    version = subprocess.run([args.katago_bin, "version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    provenance = {
        "max_visits": args.max_visits, "engine_version": version,
        "engine_model_sha256": sha256(args.katago_model),
        "engine_config_sha256": sha256(args.katago_config),
        "source_dataset_sha256": sha256(args.dataset),
    }
    cfg = KataGoConfig(
        binary_path=args.katago_bin, model_path=args.katago_model,
        config_path=args.katago_config, board_size=first_chunk[0][1].size,
        rules=args.relabel_rules, komi=args.relabel_komi, request_timeout_seconds=120,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    engine = KataGoAnalysisClient(cfg)
    started = time.perf_counter()
    written = 0
    games, splits, phases = set(), Counter(), Counter()
    try:
        engine.start()
        with temporary.open("w") as handle:
            chunk = first_chunk
            first_chunk = None
            while chunk:
                results = engine.analyze_positions(
                    [board for _, board in chunk], max_visits=args.max_visits,
                    include_policy=True, rules=args.relabel_rules, komi=args.relabel_komi,
                )
                for (record, board), result in zip(chunk, results, strict=True):
                    teacher = make_teacher_record(
                        record, board, result, rules=args.relabel_rules, komi=args.relabel_komi,
                        prior_mix=args.prior_mix, provenance=provenance,
                    )
                    handle.write(json.dumps(teacher) + "\n")
                    written += 1
                    games.add(record["game_id"])
                    splits[record["split"]] += 1
                    phases[move_phase(record["move_number"])] += 1
                handle.flush()
                print(json.dumps({"positions": written, "selected": len(records),
                                  "rejected": len(rejected), "seconds": time.perf_counter() - started}), flush=True)
                # Drop cached histories before constructing the next batch.
                chunk.clear()
                chunk = next(batches, None)
    finally:
        engine.close()
    temporary.replace(destination)
    manifest = {
        "arguments": vars(args), "teacher": provenance, "written": written,
        "games": len(games), "splits": dict(splits), "move_phases": dict(phases),
        "terminal_filter": "exclude any history reaching two consecutive passes",
        "rejected": rejected, "seconds": time.perf_counter() - started,
        "output_sha256": sha256(destination),
        "warning": "Approximate teacher labels at the stated budget; source game outcomes do not apply after relabeling.",
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
