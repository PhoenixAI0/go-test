"""Compare actual moves using equal-budget KataGo child-position estimates.

Example:
  uv run python scripts/eval_move_quality.py --dataset DATA_VALIDATION \
    --train-data DATA_TRAIN --policy-checkpoint policy=CHECKPOINT \
    --output REPORT.json --katago-model NETWORK --katago-config CONFIG

Repeat --policy-checkpoint LABEL=PATH or --text-model LABEL=PATH to compare
models on identical held-out states. A seeded uniform legal baseline is always
included. Results are approximate single-move diagnostics, not full-game Elo.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from go_rl.config import KataGoConfig
from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256, read_records, select_game_balanced, validate_splits
from go_rl.katago import KataGoAnalysisClient
from go_rl.indexed_jsonl import IndexedJsonlRecords
from go_rl.model_interface import TextGoPolicy, action_to_move, load_text_model
from go_rl.move_quality import (
    filter_preterminal_records, score_move_predictions, summarize_move_quality, terminal_history,
)
from go_rl.policy_model import load_policy_model
from go_rl.policy_inference import infer_policy_batch
from go_rl.static_policy import fit_response_frequencies, frequency_move


def checkpoint_provenance(path: str) -> dict:
    directory = Path(path)
    if not directory.is_dir():
        return {"source": path, "local_file_sha256": None}
    # Checkpoints are small adapters/heads; include tokenizer and architecture metadata.
    return {"source": path, "local_file_sha256": {
        str(file.relative_to(directory)): file_sha256(file)
        for file in sorted(directory.rglob("*")) if file.is_file()
    }}


@torch.inference_mode()
def predict_categorical(path: str, records: list[dict], args, symmetries: int = 1) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    model, tokenizer, metadata = load_policy_model(path, args.device)
    encoding = args.encoding or metadata.get("encoding") or "compact"
    predictions = []
    token_count = 0
    forward_seconds = 0.
    forward_sequences = 0
    forward_calls = 0
    for offset in range(0, len(records), args.batch_size):
        chunk = records[offset:offset + args.batch_size]
        outputs = infer_policy_batch(model, tokenizer, chunk, encoding=encoding, symmetries=symmetries)
        forward_seconds += outputs.forward_seconds
        forward_sequences += outputs.forward_sequences
        forward_calls += outputs.forward_calls
        actions = outputs.probabilities.argmax(-1).tolist()
        values = outputs.value.tolist()
        for action, value, tokens, lengths in zip(actions, values, outputs.prompt_tokens,
                                                  outputs.orientation_prompt_tokens, strict=True):
            predictions.append({"move": action_to_move(action, model.board_size), "model_value_estimate": value,
                                "prompt_tokens": tokens, "orientation_prompt_tokens": lengths,
                                "decoding": "masked_greedy_categorical" if symmetries == 1
                                            else "d4_mean_legal_probability_argmax"})
            token_count += tokens
    return predictions, {**checkpoint_provenance(path), "checkpoint_metadata": metadata,
                          "encoding": encoding, "seconds_with_loading": time.perf_counter() - started,
                          "forward_seconds": forward_seconds, "prompt_tokens": token_count,
                          "source_batch_size": args.batch_size, "symmetries": symmetries,
                          "forward_sequences": forward_sequences, "forward_calls": forward_calls,
                          "maximum_forward_batch_size": args.batch_size * symmetries}


def predict_text(path: str, records: list[dict], args, num_beams: int = 1) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    model, tokenizer = load_text_model(path, args.device, args.revision)
    metadata_path = Path(path) / "go_text.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    encoding = args.encoding or metadata.get("encoding") or "compact"
    policy = TextGoPolicy(model, tokenizer, encoding=encoding, constrained=not args.text_unconstrained,
                          thinking=False, temperature=0., max_new_tokens=args.max_new_tokens,
                          num_beams=num_beams)
    predictions = [asdict(policy.predict(board_from_example(record), rules=record["rules"],
                                        komi=float(record["komi"]))) for record in records]
    return predictions, {**checkpoint_provenance(path), "checkpoint_metadata": metadata,
                          "base_model": getattr(model.config, "_name_or_path", None),
                          "resolved_revision": getattr(model.config, "_commit_hash", None),
                          "encoding": encoding, "seconds_with_loading": time.perf_counter() - started,
                          "forward_seconds": sum(row["seconds"] for row in predictions),
                          "batch_size": 1, "constrained": not args.text_unconstrained,
                          "num_beams": num_beams,
                          "decoding_objective": ("raw_complete_string_log_probability_including_eos"
                                                 if num_beams > 1 else "greedy_token_prefix")}


def predict_frequency(path: str, records: list[dict], args) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    size = board_from_example(records[0]).size
    counts, source_rows = fit_response_frequencies(
        IndexedJsonlRecords(path), board_size=size, symmetries=args.frequency_symmetries)
    predictions = [{"move": frequency_move(counts, board_from_example(record), rules=record["rules"]),
                    "decoding": "empirical_teacher_response_frequency_legal_argmax"} for record in records]
    return predictions, {"source": path, "source_sha256": file_sha256(path), "source_rows": source_rows,
                          "source_split": "train", "label": "response", "symmetries": args.frequency_symmetries,
                          "action_counts": counts, "tie_break": "row_major_then_pass",
                          "terminal_source_rows_retained_to_match_training_distribution": sum(
                              terminal_history(board_from_example(row)) for row in IndexedJsonlRecords(path)),
                          "seconds_with_loading": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-data", action="append", required=True,
                        help="Training source for game/state leakage audit; repeat for checkpoints with different data")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--policy-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--policy-symmetries", type=int, nargs="+", choices=(1, 8), default=[1],
                        help="Compare categorical direct/D4-averaged probabilities; expands each forward batch by S")
    parser.add_argument("--text-model", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--frequency-baseline", action="append", default=[], metavar="LABEL=TRAIN_JSONL")
    parser.add_argument("--frequency-symmetries", type=int, choices=(1, 8), default=1,
                        help="Expand training response counts by D4; no held-out labels are used")
    parser.add_argument("--revision", default=None, help="Pinned revision for a text hub model")
    parser.add_argument("--encoding", choices=("legacy", "grid", "compact", "stones", "cells"), default=None)
    parser.add_argument("--text-unconstrained", action="store_true")
    parser.add_argument("--text-beam-widths", type=int, nargs="+", default=[1],
                        help="Compare text decoding widths on identical child estimates; 1=greedy")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-batch-size", type=int, default=8)
    parser.add_argument("--max-visits", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", required=True)
    parser.add_argument("--katago-bin", default=os.environ.get("KATAGO_BIN", "katago"))
    parser.add_argument("--katago-model", default=os.environ.get("KATAGO_MODEL"), required=not os.environ.get("KATAGO_MODEL"))
    parser.add_argument("--katago-config", required=True)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.max_positions, args.batch_size, args.analysis_batch_size,
           args.max_visits, args.bootstrap_samples, *args.text_beam_widths) < 1:
        parser.error("All position, token, batch, visit and bootstrap budgets must be positive")
    args.text_beam_widths = list(dict.fromkeys(args.text_beam_widths))
    args.policy_symmetries = list(dict.fromkeys(args.policy_symmetries))
    if args.text_unconstrained and any(width > 1 for width in args.text_beam_widths):
        parser.error("Beam move decoding requires constrained text output")
    sources = []
    names = {"uniform", "reference"}
    for kind, entries in (("categorical", args.policy_checkpoint), ("text", args.text_model),
                          ("frequency", args.frequency_baseline)):
        for entry in entries:
            name, separator, path = entry.partition("=")
            if not separator or not name or not path or name in names:
                parser.error("Each model needs a unique nonempty LABEL=PATH; uniform/reference are reserved")
            widths = (args.text_beam_widths if kind == "text" else
                      args.policy_symmetries if kind == "categorical" else [1])
            for width in widths:
                label = f"{name}/beam{width}" if kind == "text" and (len(widths) > 1 or width > 1) else name
                if kind == "categorical" and (len(widths) > 1 or width > 1):
                    label = f"{name}/sym{width}"
                if label in names:
                    parser.error(f"Duplicate generated model label: {label}")
                names.add(label)
                sources.append((kind, label, path, width))
    if not sources:
        parser.error("Supply at least one categorical/text model or fitted frequency baseline")
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a fresh path to preserve provenance")
    records = [record for record in read_records(args.dataset) if record.get("split") == args.split]
    if not records:
        parser.error("No positions have the requested held-out split")
    # Audit every training source against the same held-out set without keeping
    # complete training-game histories resident merely to check isolation.
    training_sources = sorted(set(args.train_data) | {path for kind, _, path, _ in sources if kind == "frequency"})
    for source in training_sources:
        validate_splits(IndexedJsonlRecords(source), records)
    source_positions = len(records)
    records, rejected_positions = filter_preterminal_records(records)
    if not records:
        parser.error("No preterminal held-out positions remain")
    records = select_game_balanced(records, args.max_positions, args.seed)
    sizes = {board_from_example(record).size for record in records}
    if len(sizes) != 1:
        parser.error("Use one board size per evaluation")
    torch.manual_seed(args.seed)
    predictions, model_provenance = {}, {}
    started = time.perf_counter()
    for kind, name, path, width in sources:
        if kind == "categorical":
            predictions[name], model_provenance[name] = predict_categorical(path, records, args, symmetries=width)
        elif kind == "frequency":
            predictions[name], model_provenance[name] = predict_frequency(path, records, args)
        else:
            predictions[name], model_provenance[name] = predict_text(path, records, args, num_beams=width)
        model_provenance[name]["kind"] = kind
        print(json.dumps({"event": "predictions", "model": name, "positions": len(records),
                          "seconds": model_provenance[name]["seconds_with_loading"]}), flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    version = subprocess.run([args.katago_bin, "version"], capture_output=True, text=True,
                             check=True, timeout=10).stdout.strip()
    engine = KataGoAnalysisClient(KataGoConfig(
        binary_path=args.katago_bin, model_path=args.katago_model, config_path=args.katago_config,
        board_size=sizes.pop(), request_timeout_seconds=120,
    ))
    try:
        engine.start()
        rows, query_counts = score_move_predictions(records, predictions, engine, max_visits=args.max_visits,
                                                     batch_size=args.analysis_batch_size, seed=args.seed)
    finally:
        engine.close()
    result = {
        "schema_version": 2, "arguments": vars(args), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
        "provenance": {"dataset_sha256": file_sha256(args.dataset),
                       "training_sources_sha256": {source: file_sha256(source) for source in training_sources},
                       "models": model_provenance, "katago_version": version,
                       "katago_model_sha256": file_sha256(args.katago_model),
                       "katago_config_sha256": file_sha256(args.katago_config)},
        "positions": len(rows), "games": len({row["game_id"] for row in rows}),
        "position_selection": {"source_split_positions": source_positions,
                               "excluded_before_sampling": rejected_positions,
                               "excluded_count": len(rejected_positions),
                               "sampling": "deterministic_game_balanced_after_terminal_history_filter"},
        "summary": summarize_move_quality(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed),
        "queries": query_counts, "details": rows, "seconds": time.perf_counter() - started,
        "interpretation": {
            "reference": "Fresh root search best move; its child and every model/uniform child analyzed at equal visit caps.",
            "differences": "Chosen child minus reference/uniform child, in the original player's perspective; higher is better.",
            "quality_conditioning": "Score/winrate summaries include only legal predictions; legality includes every position.",
            "confidence_intervals": "Percentile game-cluster bootstrap of equal-game means; excludes engine error and training-seed variance.",
            "isolation": "Held-out game/state separation audited against supplied --train-data; model pretraining or undisclosed training sources are outside this audit.",
            "limitations": "Engine estimates, not exact regret or verified outcomes. Uniform is one seeded legal draw per position. No match Elo.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": result["summary"], "queries": query_counts}), flush=True)


if __name__ == "__main__":
    main()
