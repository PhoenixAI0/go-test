"""Compare text interfaces on identical, game-balanced held-out Go positions.

Source-move agreement is an imitation proxy, not playing strength. Teacher regret
is measured only when the chosen move was evaluated, and coverage is reported.
Run with ``uv run python scripts/benchmark_interfaces.py --help``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
from typing import Any, Mapping, Sequence

from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import position_key, select_game_balanced, stable_digest
from go_rl.model_interface import ENCODINGS, TextGoPolicy, load_text_model
from go_rl.rules import legality_superko


def teacher_metrics(record: Mapping[str, Any], move: str, player: str) -> dict[str, Any]:
    """Compare only evaluated actions, converting declared BLACK perspective.

    The reference maximizes teacher winrate among evaluated candidates. It is
    not necessarily the visit-count best move or the best of all legal actions.
    Unsearched model moves have unknown regret, never zero or worst-case regret.
    """
    infos = record.get("teacher_move_infos")
    if not infos:
        return {"teacher_evaluation_available": False, "teacher_chosen_move_evaluated": False}
    metadata = record.get("teacher", {})
    perspective = metadata.get("perspective", metadata.get("value_perspective"))
    if perspective != "black":
        raise ValueError("teacher_move_infos require teacher.perspective='black'")
    values = {}
    for info in infos:
        if info.get("winrate") is None:
            continue
        winrate = float(info["winrate"])
        if not math.isfinite(winrate) or not 0 <= winrate <= 1:
            raise ValueError("Teacher winrates must be finite probabilities")
        score = info.get("scoreLead")
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Teacher scores must be finite")
        values[info["move"]] = (winrate if player == "B" else 1 - winrate,
                                score if player == "B" or score is None else -score)
    metrics: dict[str, Any] = {
        "teacher_evaluation_available": bool(values),
        "teacher_chosen_move_evaluated": move in values,
        "teacher_best_move_agreement": move == record.get("teacher_best_move"),
    }
    if move not in values:
        return metrics
    best_move = max(values, key=lambda candidate: (values[candidate][0], candidate))
    best_value, best_score = values[best_move]
    chosen_value, chosen_score = values[move]
    metrics.update(teacher_regret_reference_move=best_move,
                   teacher_winrate_regret_points=100 * (best_value - chosen_value))
    if best_score is not None and chosen_score is not None:
        metrics["teacher_score_loss_points"] = best_score - chosen_score
    return metrics


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in results:
        variant = f"{record['encoding']}/{record['decoding']}"
        groups[variant].append(record)
    summary = {}
    for variant, records in groups.items():
        completed = [record for record in records if "error" not in record]
        metrics: dict[str, Any] = {
            "positions": len(records), "games": len({row["game_id"] for row in records}),
            "errors": len(records) - len(completed),
            "legal_rate": sum(bool(row.get("legal")) for row in records) / len(records),
            "teacher_regret_coverage": sum(bool(row.get("teacher_chosen_move_evaluated"))
                                           for row in records) / len(records),
        }
        for name in ("source_move_agreement", "teacher_best_move_agreement", "truncated",
                     "prompt_tokens", "completion_tokens", "seconds", "teacher_winrate_regret_points",
                     "teacher_score_loss_points"):
            values = [row[name] for row in completed if name in row and row[name] is not None]
            if values:
                metrics[f"mean_{name}"] = statistics.mean(values)
                metrics[f"n_{name}"] = len(values)
        if completed:
            metrics["total_generation_seconds"] = sum(row["seconds"] for row in completed)
        summary[variant] = metrics
    return summary


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/interface_benchmark.json"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--split", choices=["train", "validation", "val", "test", "all"], default="validation")
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encodings", choices=ENCODINGS, nargs="+", default=list(ENCODINGS))
    parser.add_argument("--decodings", choices=["unconstrained", "constrained", "thinking"], nargs="+",
                        default=["unconstrained", "constrained"])
    parser.add_argument("--include-thinking", action="store_true", help="Also evaluate free reasoning with a bounded budget")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--thinking-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[1],
                        help="Constrained text widths to compare; other decoding modes keep width 1")
    args = parser.parse_args()
    if args.positions < 1 or args.max_new_tokens < 1 or args.thinking_tokens < 1:
        parser.error("Position and token budgets must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Temperature must be finite and nonnegative")
    args.beam_widths = list(dict.fromkeys(args.beam_widths))
    if min(args.beam_widths) < 1 or (max(args.beam_widths) > 1 and args.temperature > 0):
        parser.error("Beam widths must be positive; beam decoding requires temperature zero")
    split = "validation" if args.split == "val" else args.split
    with args.dataset.open("rb") as handle:
        source_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    excluded: Counter[str] = Counter()
    eligible = []
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and row.get("split") != split:
                excluded["different_or_missing_split"] += 1
                continue
            if row.get("symmetry", 0) != 0:
                excluded["augmented_duplicate"] += 1
                continue
            try:
                legality_superko(row.get("rules"))
                if row.get("komi") is None or not math.isfinite(float(row["komi"])):
                    raise ValueError("Missing or nonfinite komi")
            except ValueError as exc:
                excluded[str(exc)] += 1
                continue
            eligible.append(row)
    selected = select_game_balanced(eligible, args.positions, args.seed)
    if not selected:
        parser.error(f"No supported positions in requested split. Exclusions: {dict(excluded)}")
    boards = [board_from_example(row) for row in selected]
    encodings = list(dict.fromkeys(args.encodings))
    decodings = list(dict.fromkeys([*args.decodings, *(["thinking"] if args.include_thinking else [])]))

    import torch
    from transformers import set_seed

    set_seed(args.seed)
    model, tokenizer = load_text_model(args.model, args.device, args.revision)
    device = model.device.type
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dataset": str(args.dataset.resolve()), "dataset_sha256": source_sha,
        "seed": args.seed, "split": split, "requested_positions": args.positions,
        "selected_positions": len(selected), "eligible_positions": len(eligible),
        "excluded": dict(excluded), "encodings": encodings, "decodings": decodings,
        "max_new_tokens": args.max_new_tokens, "thinking_tokens": args.thinking_tokens,
        "temperature": args.temperature, "beam_widths": args.beam_widths,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "device": device,
                     "accelerator": torch.cuda.get_device_name() if device == "cuda" else platform.processor()},
        "versions": {package: importlib.metadata.version(package) for package in ("torch", "transformers")},
        "interpretation": "Source-move agreement is a weak imitation proxy. Teacher regret is conditional on evaluated-move coverage. These are not Elo estimates.",
    }
    results = []
    report: dict[str, Any] = {"metadata": metadata, "status": "running", "results": results, "summary": {}}
    _write_report(args.output, report)
    decoding_variants = [(decoding, width) for decoding in decodings
                         for width in (args.beam_widths if decoding == "constrained" else [1])]
    for encoding in encodings:
        for decoding, width in decoding_variants:
            for row, board in zip(selected, boards):
                # Reset per-position randomness so changes in variant order do
                # not consume a different RNG stream for sampled comparisons.
                set_seed(int(stable_digest([args.seed, position_key(row)])[:8], 16))
                result = {"position_key": position_key(row), "game_id": row["game_id"],
                          "move_number": row.get("move_number"), "encoding": encoding,
                          "decoding": f"{decoding}_beam{width}" if width > 1 else decoding,
                          "num_beams": width,
                          "rules": row["rules"], "komi": row["komi"], "to_play": board.next_player(),
                          "source_response": row.get("source_response", row.get("response"))}
                policy = TextGoPolicy(model, tokenizer, encoding=encoding,
                                      constrained=decoding == "constrained", thinking=decoding == "thinking",
                                      max_new_tokens=args.thinking_tokens if decoding == "thinking" else args.max_new_tokens,
                                      temperature=args.temperature, superko=legality_superko(row["rules"]),
                                      num_beams=width)
                try:
                    prediction = policy.predict(board, komi=float(row["komi"]), rules=row["rules"])
                    result.update(asdict(prediction))
                    if result["source_response"] is not None:
                        result["source_move_agreement"] = prediction.move == result["source_response"]
                    result.update(teacher_metrics(row, prediction.move, board.next_player()))
                except ValueError as exc:
                    result["error"] = str(exc)
                results.append(result)
                report["summary"] = summarize(results)
                _write_report(args.output, report)
                print(f"{encoding}/{decoding} {len(results)}: move={result.get('move')} legal={result.get('legal')} error={result.get('error')}", flush=True)
    report["status"] = "complete"
    _write_report(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
