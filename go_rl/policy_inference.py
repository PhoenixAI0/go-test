"""Batched categorical inference with optional exact D4 geometry averaging.

This defines an inference policy: average orientation-specific legal-action
probabilities, not logits. It is not the behavior policy used by PPO rollouts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
import time
from typing import Any

import torch
from torch import Tensor

from .data_prep import (
    _transform_coords, board_from_example, state_record_from_board, transform_teacher_example,
)
from .model_interface import legal_action_mask, synchronize_device
from .policy_model import collate_policy_inputs, encode_policy_position
from .rules import legality_superko
from .teacher_data import validate_history_for_rules


@lru_cache(maxsize=144)
def action_permutation(board_size: int, symmetry: int) -> tuple[int, ...]:
    """Map original row-major actions to their transformed indices; pass is fixed."""
    if type(board_size) is not int or not 2 <= board_size <= 19:
        raise ValueError("board_size must be an integer between 2 and 19")
    if type(symmetry) is not int or symmetry not in range(8):
        raise ValueError("symmetry must be an integer in [0,7]")
    destinations = []
    for action in range(board_size * board_size):
        row, column = _transform_coords(divmod(action, board_size), board_size, symmetry)
        destinations.append(row * board_size + column)
    return (*destinations, board_size * board_size)


@dataclass(frozen=True)
class PolicyInferenceOutput:
    probabilities: Tensor
    value: Tensor
    prompt_tokens: list[int]
    orientation_prompt_tokens: list[list[int]]
    symmetries: int
    forward_seconds: float
    forward_sequences: int
    forward_calls: int = 1


@torch.inference_mode()
def infer_policy_batch(
    model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], *,
    encoding: str = "compact", symmetries: int = 1,
) -> PolicyInferenceOutput:
    """Use one forward for B×S complete observations; return B original-frame rows.

    Legality is computed once per original state, including known superko
    history, then permuted exactly. Each transformed encoding retains setup and
    move history, so recent-move context follows the same geometric transform.
    Inputs, labels, model parameters and model train/eval mode are not mutated.
    ``prompt_tokens`` counts actual tokens across every orientation, not just
    the source orientation. Caller batch size therefore expands by ``S``.
    """
    if type(symmetries) is not int or symmetries not in (1, 8):
        raise ValueError("Policy symmetries must be 1 or 8")
    if not records:
        raise ValueError("Policy inference needs at least one position")
    if model.training:
        raise ValueError("Policy inference requires model.eval()")
    if getattr(model, "readout", "last") in {"cells", "cell_attention"} and encoding != "cells":
        raise ValueError("Spatial policy readout requires the cells encoding")
    size = model.board_size
    permutations = [action_permutation(size, symmetry) for symmetry in range(symmetries)]
    context = getattr(model.backbone.config, "max_position_embeddings", float("inf"))
    sequences, masks, lengths, original_masks = [], [], [], []
    for record in records:
        board = board_from_example(record)
        if board.size != size:
            raise ValueError("Evaluation board size differs from categorical checkpoint")
        komi = float(record["komi"])
        if not math.isfinite(komi):
            raise ValueError("Komi must be finite")
        validate_history_for_rules(board, record["rules"])
        legal = legal_action_mask(board, superko=legality_superko(record["rules"]))
        original_masks.append(legal)
        # Do not copy/transform teacher distributions or outcomes into inference.
        state = state_record_from_board(board) if symmetries > 1 else None
        position_lengths = []
        for symmetry, permutation in enumerate(permutations):
            transformed = board if symmetry == 0 else board_from_example(transform_teacher_example(state, symmetry))
            ids = encode_policy_position(transformed, tokenizer, encoding,
                                         rules=record["rules"], komi=komi)
            if len(ids) > context:
                raise ValueError("Position exceeds model context; refusing to truncate")
            sequences.append(ids)
            position_lengths.append(len(ids))
            mask = [False] * len(legal)
            for action, destination in enumerate(permutation):
                mask[destination] = legal[action]
            masks.append(mask)
        lengths.append(position_lengths)
    legal_tensor = torch.tensor(masks, dtype=torch.bool, device=model.device)
    inputs = collate_policy_inputs(sequences, tokenizer, model.device)
    synchronize_device(torch.device(model.device).type)
    started = time.perf_counter()
    outputs = model(**inputs, legal_mask=legal_tensor)
    synchronize_device(torch.device(model.device).type)
    elapsed = time.perf_counter() - started
    if outputs.logits.shape != legal_tensor.shape or outputs.value.shape != (len(sequences),):
        raise ValueError("Model returned inconsistent policy/value shapes")
    if (not bool(torch.isfinite(outputs.logits[legal_tensor]).all())
            or not bool(torch.isfinite(outputs.value).all()) or bool((outputs.value.abs() > 1).any())):
        raise ValueError("Model returned invalid policy/value predictions")
    probabilities = outputs.logits.float().masked_fill(~legal_tensor, float("-inf")).softmax(-1)
    probabilities = probabilities.reshape(len(records), symmetries, -1)
    # gather[t(a)] assigns transformed probability t(a) back to original a.
    inverse_gather = torch.tensor(permutations, dtype=torch.long, device=model.device)
    probabilities = probabilities.gather(2, inverse_gather.unsqueeze(0).expand(len(records), -1, -1)).mean(1)
    original_legal = torch.tensor(original_masks, dtype=torch.bool, device=model.device)
    probabilities = probabilities.masked_fill(~original_legal, 0.)
    # Average probabilities already sum to one; normalize only rounding drift.
    probabilities /= probabilities.sum(-1, keepdim=True)
    values = outputs.value.float().reshape(len(records), symmetries).mean(1)
    return PolicyInferenceOutput(
        probabilities=probabilities, value=values, prompt_tokens=[sum(row) for row in lengths],
        orientation_prompt_tokens=lengths, symmetries=symmetries, forward_seconds=elapsed,
        forward_sequences=len(sequences),
    )
