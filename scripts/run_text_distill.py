"""Bounded Qwen text-move SFT, comparable with run_policy_distill.py.

Only completion positions are projected into the vocabulary. Exact non-thinking
chat prefixes match TextGoPolicy; validation never silently truncates a board.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import chain
import json
import math
from pathlib import Path
import platform
import random
import time

import torch

from go_rl.data_prep import board_from_example, transform_teacher_example
from go_rl.experiment_utils import file_sha256, select_game_balanced_indices, validate_splits
from go_rl.indexed_jsonl import IndexedJsonlRecords, iter_augmented_records, write_record_cache
from go_rl.model_interface import ENCODINGS, TextGoPolicy, choose_device, load_text_model, synchronize_device
from go_rl.rules import legality_superko
from go_rl.text_distill import (
    collate_completions, completion_loss, completion_losses, create_text_model,
    prepare_text_record, save_text_model, uniform_objective_metadata,
)
from go_rl.training_checkpoint import (
    preserve_best_snapshot, restore_best_snapshot, restore_training_state, save_training_state,
)


@torch.inference_mode()
def evaluate_nll(model, tokenizer, records: list[dict], batch_size: int) -> dict:
    was_training = model.training
    model.eval()
    total_nll, total_tokens, total_sequences = 0., 0, 0
    synchronize_device(model.device.type)
    started = time.perf_counter()
    for offset in range(0, len(records), batch_size):
        chunk = records[offset:offset + batch_size]
        batch = collate_completions(chunk, tokenizer.pad_token_id, model.device)
        losses, mask = completion_losses(model, batch)
        total_nll += losses.sum().item()
        total_tokens += mask.sum().item()
        total_sequences += len(chunk)
    synchronize_device(model.device.type)
    model.train(was_training)
    return {"completion_token_nll": total_nll / total_tokens,
            "completion_sequence_nll": total_nll / total_sequences,
            "positions": total_sequences, "completion_tokens": total_tokens,
            "seconds": time.perf_counter() - started}


def evaluate_generation(model, tokenizer, records: list[dict], encoding: str, max_new_tokens: int) -> dict:
    was_training = model.training
    model.eval()
    predictions = []
    for record in records:
        board = board_from_example(record)
        result = {"game_id": record["game_id"], "move_number": record.get("move_number"),
                  "expected": record["response"]}
        for name, constrained in (("raw", False), ("constrained", True)):
            policy = TextGoPolicy(model, tokenizer, encoding=encoding, constrained=constrained,
                                  thinking=False, temperature=0., max_new_tokens=max_new_tokens,
                                  superko=legality_superko(record["rules"]))
            prediction = policy.predict(board, komi=float(record["komi"]), rules=record["rules"])
            result[name] = asdict(prediction)
        predictions.append(result)
    model.train(was_training)
    count = len(predictions)
    return {
        "positions": count,
        "raw_legal_rate": sum(row["raw"]["legal"] for row in predictions) / count,
        "raw_top1": sum(row["raw"]["move"] == row["expected"] for row in predictions) / count,
        "raw_truncated_rate": sum(row["raw"]["truncated"] for row in predictions) / count,
        "constrained_legal_rate": sum(row["constrained"]["legal"] for row in predictions) / count,
        "constrained_top1": sum(row["constrained"]["move"] == row["expected"] for row in predictions) / count,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--init-checkpoint", help="Start a new objective/budget from an existing text adapter")
    parser.add_argument("--resume-from", help="Resume the same planned run from a model+optimizer checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--encoding", choices=ENCODINGS, default="compact")
    parser.add_argument("--symmetries", type=int, choices=(1, 8), default=1,
                        help="D4 augmentation of training examples only; split validation first")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--generation-positions", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.init_checkpoint and args.resume_from:
        parser.error("Choose init-checkpoint for a new experiment or resume-from for exact continuation")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every must be nonnegative")
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.eval_every,
           args.generation_positions, args.max_new_tokens, args.max_seq_length) < 1:
        parser.error("Step, batch, evaluation and token budgets must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; use a fresh experiment path")
    train_records, eval_records = IndexedJsonlRecords(args.train_data), IndexedJsonlRecords(args.eval_data)
    validate_splits(train_records, eval_records)
    objective_metadata = uniform_objective_metadata(chain(train_records, eval_records))
    source_train_positions = len(train_records)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    if checkpoint := args.resume_from or args.init_checkpoint:
        metadata = json.loads((Path(checkpoint) / "go_text.json").read_text())
        if metadata.get("encoding") != args.encoding:
            parser.error("Checkpoint encoding differs; start a fresh model for this ablation")
        if not (Path(checkpoint) / "adapter_config.json").is_file():
            parser.error("A text distillation checkpoint must contain a PEFT adapter")
        model, tokenizer = load_text_model(checkpoint, device, trainable=True)
    else:
        model, tokenizer, metadata = create_text_model(
            args.model, device=device, revision=args.revision,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    metadata.update(objective_metadata, encoding=args.encoding)
    max_context = min(args.max_seq_length, model.config.max_position_embeddings)
    augmented = iter_augmented_records(train_records, args.symmetries, transform_teacher_example)
    train = write_record_cache(
        (prepare_text_record(row, tokenizer, args.encoding, max_seq_length=max_context) for row in augmented),
        output / "prepared_train.jsonl")
    validation = write_record_cache(
        (prepare_text_record(row, tokenizer, args.encoding, max_seq_length=max_context) for row in eval_records),
        output / "prepared_validation.jsonl")
    generation_records = eval_records.select(
        select_game_balanced_indices(eval_records, args.generation_positions, args.seed))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    warmup = max(1, min(10, args.steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.) * .5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)))
    rng = random.Random(args.seed)
    run = {"arguments": vars(args), "model": metadata, "platform": platform.platform(),
           "torch": torch.__version__, "device": device,
           "train_sha256": file_sha256(args.train_data), "eval_sha256": file_sha256(args.eval_data),
           "source_train_positions": source_train_positions, "training_symmetries": args.symmetries,
           "prepared_data": {"storage": "indexed_jsonl", "train_bytes": train.path.stat().st_size,
                             "eval_bytes": validation.path.stat().st_size,
                             "train_sha256": file_sha256(train.path), "eval_sha256": file_sha256(validation.path),
                             "index_bytes": train.index_bytes + validation.index_bytes},
           "train_positions": len(train), "eval_positions": len(validation),
           "train_games": len({row["game_id"] for row in train}),
           "eval_games": len({row["game_id"] for row in validation}),
           "mean_prompt_tokens": sum(len(row["prompt_ids"]) for row in train) / len(train),
           "mean_answer_tokens": sum(len(row["answer_ids"]) for row in train) / len(train),
           "trainable_parameters": sum(parameter.numel() for parameter in parameters)}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    contract = {name: getattr(args, name) for name in (
        "steps", "batch_size", "gradient_accumulation", "learning_rate", "encoding", "symmetries",
        "seed", "max_seq_length", "eval_every", "generation_positions", "max_new_tokens")}
    contract.update(train_sha256=run["train_sha256"], eval_sha256=run["eval_sha256"],
                    train_prepared_sha256=run["prepared_data"]["train_sha256"],
                    eval_prepared_sha256=run["prepared_data"]["eval_sha256"],
                    torch=str(torch.__version__), trainer_version=2)
    first_step = restore_training_state(args.resume_from, optimizer, scheduler, rng,
                                       contract=contract, device=device) if args.resume_from else 0
    if first_step >= args.steps:
        parser.error("Checkpoint already reached the planned steps; use init-checkpoint for a new training budget")
    started = time.perf_counter()
    baseline = evaluate_nll(model, tokenizer, validation, args.batch_size)
    baseline_generation = evaluate_generation(model, tokenizer, generation_records, args.encoding, args.max_new_tokens)
    (output / "generation_before.json").write_text(json.dumps(baseline_generation, indent=2) + "\n")
    best = baseline["completion_token_nll"]
    restored_best = bool(args.resume_from and restore_best_snapshot(args.resume_from, output / "best"))
    if restored_best:
        previous_best = json.loads((output / "best" / "go_text.json").read_text())["validation"]["completion_token_nll"]
        best = min(best, previous_best)
    if not restored_best or baseline["completion_token_nll"] <= best:
        save_text_model(model, tokenizer, output / "best", {**metadata, "step": first_step, "validation": baseline})
    entry = {"step": first_step, "eval": baseline, "elapsed_seconds": time.perf_counter() - started}
    print(json.dumps(entry), flush=True)
    with (output / "metrics.jsonl").open("w") as log:
        log.write(json.dumps(entry) + "\n")
        log.flush()
        model.train()
        for step in range(first_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.
            for _ in range(args.gradient_accumulation):
                chunk = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
                batch = collate_completions(chunk, tokenizer.pad_token_id, device)
                loss = completion_loss(model, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite completion training loss")
                (loss / args.gradient_accumulation).backward()
                step_loss += loss.detach().item() / args.gradient_accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Nonfinite completion gradient norm")
            optimizer.step()
            scheduler.step()
            entry = {"step": step, "train_loss": step_loss, "gradient_norm": gradient_norm.item(),
                     "elapsed_seconds": time.perf_counter() - started}
            if step % args.eval_every == 0 or step == args.steps:
                entry["eval"] = evaluate_nll(model, tokenizer, validation, args.batch_size)
                if entry["eval"]["completion_token_nll"] < best:
                    best = entry["eval"]["completion_token_nll"]
                    save_text_model(model, tokenizer, output / "best",
                                    {**metadata, "step": step, "validation": entry["eval"]})
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                snapshot = output / f"checkpoint-{step:08d}"
                save_text_model(model, tokenizer, snapshot, {**metadata, "step": step})
                preserve_best_snapshot(snapshot, output / "best")
                save_training_state(snapshot, optimizer, scheduler, rng, step=step, contract=contract, device=device)
    save_text_model(model, tokenizer, output / "final", {**metadata, "step": args.steps, "validation": entry["eval"]})
    preserve_best_snapshot(output / "final", output / "best")
    save_training_state(output / "final", optimizer, scheduler, rng, step=args.steps, contract=contract, device=device)
    final_generation = evaluate_generation(model, tokenizer, generation_records, args.encoding, args.max_new_tokens)
    (output / "generation_after.json").write_text(json.dumps(final_generation, indent=2) + "\n")
    summary = {"baseline": baseline, "final": entry["eval"], "best_validation_completion_nll": best,
               "best_scope": "full_run" if not args.resume_from or restored_best else "continuation_segment_only",
               "seconds": time.perf_counter() - started,
               "baseline_generation": {key: value for key, value in baseline_generation.items() if key != "predictions"},
               "final_generation": {key: value for key, value in final_generation.items() if key != "predictions"},
               "limitation": "Completion NLL and move agreement are imitation diagnostics, not Elo; generation subset is bounded and game-disjoint."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
