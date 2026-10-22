"""Single-pass Go policy/value heads on a pretrained language-model backbone.

This is an experimental alternative to autoregressive move text. It retains the
Qwen transformer while replacing the large vocabulary projection with a finite
action distribution. Rules mask illegal moves; no engine chooses the action.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .board import BoardState, coords_to_gtp
from .model_interface import canonical_rules_text, choose_device, inference_dtype, render_position


@dataclass
class PolicyValueOutput:
    logits: Tensor
    value: Tensor


class GoPolicyValueModel(nn.Module):
    def __init__(self, backbone: nn.Module, board_size: int = 19, *,
                 readout: str = "last", attention: str = "causal",
                 cell_suffix_tokens: int = 0, head_init_std: float = .001):
        super().__init__()
        if not 2 <= board_size <= 19:
            raise ValueError("board_size must be between 2 and 19")
        self.backbone = backbone
        self.board_size = board_size
        if readout not in {"last", "cells", "cell_attention"} or attention not in {"causal", "bidirectional"}:
            raise ValueError("Unsupported policy readout or attention mode")
        if cell_suffix_tokens < 0 or head_init_std < 0:
            raise ValueError("Cell suffix length and head initialization scale must be nonnegative")
        self.readout, self.attention = readout, attention
        self.cell_suffix_tokens = cell_suffix_tokens
        width = backbone.config.hidden_size
        # Keep the small heads in float32 even when the frozen backbone uses BF16.
        if readout == "cell_attention":
            self.policy_head = nn.ModuleDict({"query": nn.Linear(width, 128, bias=False),
                                              "key": nn.Linear(width, 128, bias=False)})
        else:
            self.policy_head = nn.Linear(width, board_size * board_size + 1 if readout == "last" else 1)
        if readout != "last":
            self.pass_head = nn.Linear(width, 1)
            nn.init.normal_(self.pass_head.weight, std=head_init_std)
            nn.init.zeros_(self.pass_head.bias)
        self.value_head = nn.Linear(width, 1)
        if readout == "cell_attention":
            # Keep a nonzero global query so the key scorer learns immediately;
            # small key weights keep the initial distribution near uniform.
            nn.init.normal_(self.policy_head["key"].weight, std=head_init_std)
        else:
            nn.init.normal_(self.policy_head.weight, std=head_init_std)
            nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def head_names(self) -> tuple[str, ...]:
        return ("policy_head", "value_head", "pass_head") if self.readout != "last" else ("policy_head", "value_head")

    def head_parameters(self):
        for name in self.head_names:
            yield from getattr(self, name).parameters()

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                legal_mask: Tensor | None = None) -> PolicyValueOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, sequence] shape")
        if not bool(attention_mask.any(dim=1).all()):
            raise ValueError("Every input must contain at least one unmasked token")
        # Position IDs make left/right padding numerically equivalent for RoPE.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        backbone_mask = attention_mask
        if self.attention == "bidirectional":
            # Transformers accepts a precomputed additive 4D mask unchanged.
            # No causal constraint is needed: the entire board is an observed
            # input, and targets/answers never appear among these tokens.
            dtype = self.backbone.get_input_embeddings().weight.dtype
            shape = (input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
            backbone_mask = torch.zeros(shape, dtype=dtype, device=input_ids.device)
            backbone_mask.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(dtype).min)
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=backbone_mask,
            position_ids=position_ids, use_cache=False, return_dict=True,
        )
        indices = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = indices.masked_fill(attention_mask == 0, -1).max(dim=1).values
        hidden = outputs.last_hidden_state[torch.arange(input_ids.shape[0], device=input_ids.device), last].float()
        if self.readout == "last":
            logits = self.policy_head(hidden)
        else:
            count = self.board_size * self.board_size
            cell_end = last - self.cell_suffix_tokens + 1
            cell_indices = cell_end[:, None] - count + torch.arange(count, device=input_ids.device)
            if bool((cell_indices < 0).any()) or not bool(attention_mask.gather(1, cell_indices).all()):
                raise ValueError("Cell readout requires a contiguous complete fixed-width board")
            cell_hidden = outputs.last_hidden_state.gather(
                1, cell_indices[:, :, None].expand(-1, -1, outputs.last_hidden_state.shape[-1])).float()
            if self.readout == "cell_attention":
                query, keys = self.policy_head["query"](hidden), self.policy_head["key"](cell_hidden)
                point_logits = (keys * query[:, None, :]).sum(-1) / query.shape[-1] ** .5
            else:
                point_logits = self.policy_head(cell_hidden).squeeze(-1)
            logits = torch.cat((point_logits, self.pass_head(hidden)), -1)
        value = self.value_head(hidden).squeeze(-1).tanh()
        if legal_mask is not None:
            if legal_mask.shape != logits.shape or legal_mask.dtype != torch.bool:
                raise ValueError("legal_mask must be a boolean [batch, actions] tensor")
            if not bool(legal_mask.any(dim=1).all()):
                raise ValueError("Every state must allow at least one action")
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return PolicyValueOutput(logits=logits, value=value)


def encode_policy_position(
    board: BoardState, tokenizer: Any, encoding: str = "compact", *,
    komi: float = 7.5, rules: str = "chinese",
) -> list[int]:
    """Encode text or a fixed-width cell sequence using existing token embeddings.

    Cells use one existing token per intersection, preventing BPE from merging
    long empty rows. This is an ablation, not a claim that token count alone
    determines quality. The complete move history is kept in the external state.
    """
    if encoding != "cells":
        prompt = render_position(board, encoding, komi=komi, rules=rules)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
    header = (
        f"Go board {board.size}x{board.size}; to play {board.next_player()}; "
        f"rules {canonical_rules_text(rules)}; komi {komi:g} for White. "
        "Cells top-to-bottom, left-to-right; X black, O white, . empty.\n"
    )
    if board.moves:
        header += "Recent moves: " + " ".join(
            f"{color}:{coords_to_gtp(coords, board.size)}" for color, coords in board.moves[-8:]) + "\n"
    ids = tokenizer.encode(header, add_special_tokens=False)
    cells = [tokenizer.encode(char, add_special_tokens=False) for char in ".XO"]
    if any(len(tokens) != 1 for tokens in cells):
        raise ValueError("Cell encoding requires single-token '.', 'X', and 'O'")
    ids.extend(cells[cell][0] for row in board.current_board() for cell in row)
    ids.extend(tokenizer.encode("\nmove:", add_special_tokens=False))
    return ids


def collate_policy_inputs(sequences: Sequence[Sequence[int]], tokenizer: Any,
                          device: str | torch.device = "cpu") -> dict[str, Tensor]:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Cannot collate empty sequences")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer must define pad or EOS")
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[index, :len(sequence)] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}


def policy_value_loss(
    outputs: PolicyValueOutput, targets: Tensor, *, value_targets: Tensor | None = None,
    value_weight: float = 0.25,
) -> tuple[Tensor, dict[str, float]]:
    """Forward-KL distillation / soft cross entropy with optional outcome targets.

    Zero target mass on an illegal action must not multiply -inf (0 * -inf is
    NaN). Positive mass on illegal actions is a dataset/rules error and rejected.
    Value targets use the current player's perspective, matching the head.
    """
    if targets.shape != outputs.logits.shape:
        raise ValueError("Policy targets must match logits")
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise ValueError("Targets must be finite nonnegative probabilities")
    if not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=1e-4):
        raise ValueError("Each target policy must sum to one")
    if bool(((targets > 0) & ~torch.isfinite(outputs.logits)).any()):
        raise ValueError("Teacher target assigns mass to an illegal action")
    log_probs = outputs.logits.log_softmax(-1)
    policy_loss = -(targets * log_probs.masked_fill(targets == 0, 0)).sum(-1).mean()
    value_loss = outputs.value.new_zeros(())
    if value_weight < 0:
        raise ValueError("value_weight must be nonnegative")
    if value_targets is not None:
        if value_targets.shape != outputs.value.shape or not bool(torch.isfinite(value_targets).all()):
            raise ValueError("Value targets must be finite and match batch shape")
        if bool((value_targets.abs() > 1).any()):
            raise ValueError("Value targets must lie in [-1, 1]")
        value_loss = (outputs.value - value_targets).square().mean()
    loss = policy_loss + value_weight * value_loss
    return loss, {"policy_loss": policy_loss.detach().item(), "value_loss": value_loss.detach().item()}


def create_policy_model(
    model_name: str = "Qwen/Qwen3-0.6B", *, board_size: int = 19,
    device: str = "auto", revision: str | None = None, lora_rank: int = 16,
    lora_alpha: int = 32, train_backbone: bool = True, readout: str = "last",
    attention: str = "causal", head_init_std: float = .001,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    device = choose_device(device)
    backbone = AutoModel.from_pretrained(
        model_name, revision=revision, dtype=inference_dtype(device), attn_implementation="sdpa",
    )
    if train_backbone:
        if lora_rank < 1:
            raise ValueError("Use a positive LoRA rank, or train_backbone=False for a head-only probe")
        backbone = get_peft_model(backbone, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
            target_modules="all-linear", bias="none", task_type="FEATURE_EXTRACTION",
        ))
    else:
        backbone.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    suffix = len(tokenizer.encode("\nmove:", add_special_tokens=False))
    model = GoPolicyValueModel(backbone, board_size, readout=readout, attention=attention,
                              cell_suffix_tokens=suffix, head_init_std=head_init_std).to(device)
    metadata = {
        "format_version": 2, "base_model": model_name,
        "encoder_version": 2,
        "revision": revision or getattr(backbone.config, "_commit_hash", None),
        "board_size": board_size, "adapter": train_backbone,
        "lora_rank": lora_rank, "lora_alpha": lora_alpha,
        "value_perspective": "side_to_move", "action_order": "top_left_row_major_then_pass",
        "readout": readout, "attention": attention, "cell_suffix_tokens": suffix,
        "head_init_std": head_init_std,
    }
    return model, tokenizer, metadata


def save_policy_model(model: GoPolicyValueModel, tokenizer: Any,
                      path: str | Path, metadata: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    # Frozen-head probes can reload their original immutable backbone by revision.
    if metadata["adapter"]:
        model.backbone.save_pretrained(destination / "backbone")
    heads = {f"{name}.{key}": value.detach().cpu().contiguous()
             for name in model.head_names
             for key, value in getattr(model, name).state_dict().items()}
    save_file(heads, destination / "heads.safetensors")
    tokenizer.save_pretrained(destination)
    (destination / "go_policy.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_policy_model(path: str | Path, device: str = "auto", *, trainable: bool = False):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    metadata = json.loads((path / "go_policy.json").read_text())
    if metadata.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported policy checkpoint format")
    device = choose_device(device)
    backbone = AutoModel.from_pretrained(
        metadata["base_model"], revision=metadata["revision"],
        dtype=inference_dtype(device), attn_implementation="sdpa",
    )
    if metadata["adapter"]:
        backbone = PeftModel.from_pretrained(backbone, path / "backbone", is_trainable=trainable)
    else:
        backbone.requires_grad_(False)
    model = GoPolicyValueModel(backbone, metadata["board_size"],
                              readout=metadata.get("readout", "last"),
                              attention=metadata.get("attention", "causal"),
                              cell_suffix_tokens=metadata.get("cell_suffix_tokens", 0),
                              head_init_std=metadata.get("head_init_std", .01))
    heads = load_file(path / "heads.safetensors")
    for name in model.head_names:
        getattr(model, name).load_state_dict({key[len(name) + 1:]: val for key, val in heads.items() if key.startswith(name + ".")})
    model.to(device).train(trainable)
    return model, AutoTokenizer.from_pretrained(path), metadata
