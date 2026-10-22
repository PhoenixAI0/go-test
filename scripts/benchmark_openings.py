"""Probe text-policy opening coverage on four explicit 19x19 observations.

These deterministic observations diagnose the from-empty pass failure; they
are not a playing-strength test or a replacement for completed matches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board
from go_rl.model_interface import TextGoPolicy, load_text_model
from go_rl.training_checkpoint import model_fingerprints


def opening_positions() -> list[tuple[str, BoardState]]:
    definitions = (
        ("empty_black", []),
        ("empty_white_after_black_pass", [("B", "PASS")]),
        ("white_after_black_D4", [("B", "D4")]),
        ("black_after_pass_white_D4", [("B", "PASS"), ("W", "D4")]),
    )
    result = []
    for label, moves in definitions:
        board = BoardState.empty(19)
        for player, move in moves:
            board.apply_move(player, move, strict=True)
        result.append((label, board))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh path")
    if args.num_beams < 1:
        parser.error("Beam width must be positive")
    sources = {}
    for entry in args.text_model:
        label, separator, path = entry.partition("=")
        if not separator or not label or not path or label in sources:
            parser.error("Models require unique nonempty LABEL=PATH")
        sources[label] = path
    rules, komi = "koPOSITIONALscoreAREAtaxNONEsui0", 7.5
    records, provenance = [], {}
    started = time.perf_counter()
    for label, path in sources.items():
        model, tokenizer = load_text_model(path, args.device, args.revision)
        metadata_path = Path(path) / "go_text.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        policy = TextGoPolicy(model, tokenizer, encoding=metadata.get("encoding", "compact"),
                              constrained=True, thinking=False, temperature=0., max_new_tokens=16,
                              num_beams=args.num_beams)
        provenance[label] = {"source": path, "checkpoint_metadata": metadata,
                             "files_sha256": model_fingerprints(Path(path)) if Path(path).is_dir() else None,
                             "resolved_revision": getattr(model.config, "_commit_hash", None)}
        for name, board in opening_positions():
            row = {"model": label, "state": name, "num_beams": args.num_beams,
                   "position": {**state_record_from_board(board), "rules": rules, "komi": komi},
                   **asdict(policy.predict(board, rules=rules, komi=komi))}
            records.append(row)
            print(json.dumps({key: row[key] for key in ("model", "state", "move", "seconds")}), flush=True)
        del policy, model, tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {"arguments": {**vars(args), "output": str(args.output)}, "provenance": provenance,
              "purpose": "fixed opening coverage diagnostic after observed full-game pass collapse",
              "engine_queries": 0, "seconds": time.perf_counter() - started, "predictions": records,
              "limitations": "Four constructed observations, not independent games or evidence of playing strength."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
