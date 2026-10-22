#!/usr/bin/env python3
"""Measure 19x19 legal-action throughput; this does not measure playing strength."""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

from go_rl.board import BoardState, coords_to_gtp


def benchmark(size: int, plies: int, seed: int, repeats: int) -> dict:
    rng = random.Random(seed)
    board = BoardState.empty(size)
    # Keep generation independent of the implementation's legal-move ordering.
    for _ in range(plies):
        candidates = [(row, col) for row in range(size) for col in range(size)
                      if board.is_basic_legal((row, col))]
        rng.shuffle(candidates)
        for coords in candidates:
            if board.is_strict_legal(board.next_player(), coords):
                board.apply_move(board.next_player(), coords, strict=True)
                break
        else:
            board.apply_move(board.next_player(), None, strict=True)
    points = [(row, col) for row in range(size) for col in range(size)]

    def uncached_scan():
        # Reconstruct from history for each candidate, reproducing the original
        # algorithm's repeated replay, without keeping an obsolete code copy.
        return [coords_to_gtp(coords, size) for coords in points
                if BoardState(size, list(board.moves)).is_strict_legal(board.next_player(), coords)]

    def cached_scan():
        return [coords_to_gtp(coords, size) for coords in points
                if board.is_strict_legal(board.next_player(), coords)]

    def bulk_scan():
        return board.legal_moves(include_pass=False)

    expected = bulk_scan()
    timings = {}
    for name, operation in (("replay_per_candidate", uncached_scan),
                             ("cached_per_candidate", cached_scan), ("bulk_legal_moves", bulk_scan)):
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            actual = operation()
            samples.append(1000 * (time.perf_counter() - started))
            if actual != expected:
                raise RuntimeError(f"Legal-action disagreement: {name}")
        timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "board_size": size, "plies": plies, "seed": seed, "repeats": repeats,
        "legal_intersections": len(expected), "timings": timings,
        "cached_speedup": timings["replay_per_candidate"]["median_ms"] / timings["cached_per_candidate"]["median_ms"],
        "bulk_speedup": timings["replay_per_candidate"]["median_ms"] / timings["bulk_legal_moves"]["median_ms"],
        "moves": [[player, coords_to_gtp(coords, size)] for player, coords in board.moves],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plies < 0 or args.repeats < 1:
        parser.error("plies must be nonnegative and repeats must be positive")
    result = benchmark(args.board_size, args.plies, args.seed, args.repeats)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "moves"}, indent=2))


if __name__ == "__main__":
    main()
