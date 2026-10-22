"""Completion-only causal move distillation without full-sequence vocab logits."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .data_prep import board_from_example
from .model_interface import choose_device, inference_dtype, render_position
from .rules import legality_superko, normalize_rules


@dataclass
class CompletionBatch:
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    targets: Tensor


def uniform_objective_metadata(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Save one validated deployment context, streaming over train and eval rows."""
    objective = None
    for record in records:
        komi = float(record["komi"])
        if not math.isfinite(komi):
            raise ValueError("Text objective komi must be finite")
        current = {"board_size": board_from_example(record).size,
                   "rules": normalize_rules(record["rules"]), "komi": komi}
        if objective is not None and current != objective:
            raise ValueError("Text training and evaluation require uniform board size, rules and komi")
        objective = current
    if objective is None:
        raise ValueError("Text objective requires nonempty records")
    return objective


def prepare_text_record(
    record: Mapping[str, Any], tokenizer: Any, encoding: str = "compact", *,
    max_seq_length: int = 1024,
) -> dict[str, Any]:
    """Use exactly the non-thinking generation prefix consumed at inference."""
    board = board_from_example(record)
    if record["response"] not in board.legal_moves(superko=legality_superko(record["rules"])):
        raise ValueError("Supervised move is illegal under the declared rules")
    prompt = render_position(board, encoding, komi=float(record["komi"]), rules=record["rules"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    answer_ids = tokenizer.encode(record["response"], add_special_tokens=False)
    if tokenizer.eos_token_id is None or not answer_ids or tokenizer.eos_token_id in answer_ids:
        raise ValueError("Move answers need nonempty tokenization and a separate EOS")
    if tokenizer.decode(answer_ids, skip_special_tokens=True) != record["response"]:
        raise ValueError("Move answer does not round-trip through tokenizer")
    answer_ids.append(tokenizer.eos_token_id)
    if not prompt_ids:
        raise ValueError("Generation prefix cannot be empty")
    if len(prompt_ids) + len(answer_ids) > max_seq_length:
        raise ValueError("Full board and answer exceed max_seq_length; refusing to truncate")
    return {"prompt_ids": prompt_ids, "answer_ids": answer_ids, "game_id": record["game_id"]}


def collate_completions(
    records: Sequence[Mapping[str, Any]], pad_token_id: int, device: str | torch.device = "cpu",
) -> CompletionBatch:
    """Left-pad predictor inputs and align variable-length answers at the right.

    For prompt P and answer A (including EOS), feed P+A[:-1]. The final |A|
    positions predict every answer token. Other rows' extra prompt positions are
    ignored when their answer is shorter than the batch maximum.
    """
    if not records or any(not row["prompt_ids"] or not row["answer_ids"] for row in records):
        raise ValueError("Every completion batch needs nonempty prompts and answers")
    inputs = [[*row["prompt_ids"], *row["answer_ids"][:-1]] for row in records]
    width = max(map(len, inputs))
    keep = max(len(row["answer_ids"]) for row in records)
    ids = torch.full((len(records), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    targets = torch.full((len(records), keep), -100, dtype=torch.long, device=device)
    for index, (row, sequence) in enumerate(zip(records, inputs)):
        ids[index, -len(sequence):] = torch.tensor(sequence, device=device)
        mask[index, -len(sequence):] = 1
        targets[index, -len(row["answer_ids"]):] = torch.tensor(row["answer_ids"], device=device)
    position_ids = mask.cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return CompletionBatch(ids, mask, position_ids, targets)


def completion_losses(model: Any, batch: CompletionBatch, *, efficient: bool = True) -> tuple[Tensor, Tensor]:
    """Return per-token NLL and its mask; no model-internal label shifting.

    Qwen's logits_to_keep projects only the relevant trailing hidden states into
    its large vocabulary. The reference path exists for numerical/gradient tests.
    """
    keep = batch.targets.shape[1]
    kwargs = {"logits_to_keep": keep} if efficient else {}
    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids, use_cache=False, return_dict=True, **kwargs)
    logits = outputs.logits[:, -keep:, :].float()
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.targets.reshape(-1),
                             ignore_index=-100, reduction="none").reshape_as(batch.targets)
    return losses, batch.targets != -100


def completion_loss(model: Any, batch: CompletionBatch) -> Tensor:
    losses, mask = completion_losses(model, batch)
    return losses.sum() / mask.sum()


def create_text_model(
    model_name: str = "Qwen/Qwen3-0.6B", *, device: str = "auto", revision: str | None = None,
    lora_rank: int = 16, lora_alpha: int = 32,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    device = choose_device(device)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=inference_dtype(device), attn_implementation="sdpa")
    resolved = revision or getattr(base.config, "_commit_hash", None)
    model = get_peft_model(base, LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules="all-linear",
        bias="none", task_type="CAUSAL_LM", revision=resolved,
    )).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    metadata = {"format_version": 1, "base_model": model_name, "revision": resolved,
                "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                "objective": "completion_only_causal_move_nll", "thinking": False}
    return model, tokenizer, metadata


def save_text_model(model: Any, tokenizer: Any, destination: str | Path, metadata: Mapping[str, Any]) -> None:
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "go_text.json").write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
