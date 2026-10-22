"""Explicit, measurable interfaces between a language model and the Go board.

The rules engine supplies legality, never move quality. Prefix constraints are
locally normalized token decoding; they are NOT the LM distribution conditioned
globally on a legal complete move. Keep that distinction when comparing policies.
Optional beam decoding instead approximately maximizes the original LM's legal
complete-string probability, including EOS, without length normalization.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .board import BoardState, coords_to_gtp, gtp_to_coords
from .prompting import build_prompt
from .reasoning import extract_final_answer
from .rollout import parse_move_answer

ENCODINGS = ("legacy", "grid", "compact", "stones")
COLUMNS = "ABCDEFGHJKLMNOPQRST"


def canonical_rules_text(rules: Any) -> str:
    from .rules import normalize_rules

    values = normalize_rules(rules)
    return (f"ko{values['ko']}score{values['scoring']}tax{values['tax']}"
            f"sui{int(values['suicide'])}button{int(values['hasButton'])}"
            f"whb{values['whiteHandicapBonus']}fpok{int(values['friendlyPassOk'])}")


def move_to_action(move: str, size: int = 19) -> int:
    """Top-left row-major intersections followed by pass (index size**2)."""
    coords = gtp_to_coords(move, size)
    return size * size if coords is None else coords[0] * size + coords[1]


def action_to_move(action: int, size: int = 19) -> str:
    if not 0 <= action <= size * size:
        raise ValueError(f"Action {action} is outside a {size}x{size} board")
    return coords_to_gtp(None if action == size * size else divmod(action, size), size)


def legal_action_mask(board: BoardState, *, superko: bool = True) -> list[bool]:
    mask = [False] * (board.size * board.size + 1)
    for move in board.legal_moves(superko=superko):
        mask[move_to_action(move, board.size)] = True
    return mask


def render_position(
    board: BoardState,
    encoding: str = "grid",
    *,
    komi: float = 7.5,
    rules: str = "chinese",
    recent_moves: int = 8,
) -> str:
    """Render current state, with fixed coordinates and explicit game context.

    ``legacy`` deliberately reproduces the old prompt for a baseline. The other
    encodings show recent moves for context, but full superko history remains in
    the rules engine. No text encoding should be mistaken for a complete verifier.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown encoding: {encoding}")
    if not math.isfinite(komi):
        raise ValueError("komi must be finite")
    if recent_moves < 0:
        raise ValueError("recent_moves must be nonnegative")
    if encoding == "legacy":
        return build_prompt(board)
    lines = [
        "Play Go. Return exactly one move in GTP coordinates (skip I), or PASS.",
        f"board_size: {board.size}", f"to_play: {board.next_player()}",
        f"rules: {canonical_rules_text(rules)}", f"komi: {komi:g} (points for White)",
        "X=Black O=White .=empty. Row 1 is at the bottom.",
    ]
    if encoding == "stones":
        for color in ("B", "W"):
            stones = [coords_to_gtp(coords, board.size)
                      for stone, coords in board.stones() if stone == color]
            lines.append(f"{color}: " + (" ".join(stones) if stones else "none"))
    else:
        sep = " " if encoding == "grid" else ""
        lines.append("columns: " + sep.join(COLUMNS[:board.size]))
        lines.append("board:")
        symbols = ".XO"
        for row, cells in enumerate(board.current_board()):
            lines.append(f"{board.size - row:2d}: " + sep.join(symbols[cell] for cell in cells))
    if recent_moves and board.moves:
        history = board.moves[-recent_moves:]
        lines.append("recent_moves: " + " ".join(
            f"{color}:{coords_to_gtp(coords, board.size)}" for color, coords in history))
    lines.append("move:")
    return "\n".join(lines)


class MoveTokenTrie:
    """Canonical legal moves plus EOS, including multi-token numbers like 19."""

    def __init__(self, tokenizer: Any, moves: Sequence[str]):
        if tokenizer.eos_token_id is None:
            raise ValueError("Constrained moves require an EOS token")
        if not moves:
            raise ValueError("At least one allowed move is required")
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.children: dict[tuple[int, ...], set[int]] = {}
        self.max_tokens = 0
        for move in moves:
            ids = tokenizer.encode(move, add_special_tokens=False)
            if not ids or self.eos_token_id in ids:
                raise ValueError(f"Invalid move tokenization: {move}")
            if tokenizer.decode(ids, skip_special_tokens=True) != move:
                raise ValueError(f"Move does not round-trip through tokenizer: {move}")
            ids = [*ids, self.eos_token_id]
            self.max_tokens = max(self.max_tokens, len(ids))
            for index, token in enumerate(ids):
                self.children.setdefault(tuple(ids[:index]), set()).add(token)

    def allowed(self, prefix: Sequence[int]) -> list[int]:
        key = tuple(prefix)
        # Transformers may keep already-finished rows in a batched generation.
        if key and key[-1] == self.eos_token_id:
            return [self.eos_token_id]
        if key not in self.children:
            raise ValueError(f"Generated prefix is outside the legal-move trie: {key}")
        return sorted(self.children[key])


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def synchronize_device(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def inference_dtype(device: str):
    import torch

    if device == "cpu":
        return torch.float32
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def load_text_model(model_name: str, device: str = "auto", revision: str | None = None, *, trainable: bool = False):
    """Load a base or PEFT adapter without CUDA-only quantization assumptions."""
    from pathlib import Path

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.peft_utils import find_adapter_config_file

    device = choose_device(device)
    adapter = PeftConfig.from_pretrained(model_name) if find_adapter_config_file(model_name) else None
    base = adapter.base_model_name_or_path if adapter else model_name
    base_revision = adapter.revision if adapter else revision
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=base_revision, dtype=inference_dtype(device), attn_implementation="sdpa",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, model_name, revision=revision, is_trainable=trainable)
    tokenizer_name = model_name
    if adapter and Path(model_name).is_dir() and not (Path(model_name) / "tokenizer_config.json").exists():
        tokenizer_name = base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision if tokenizer_name == model_name else base_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.to(device).train(trainable)
    return model, tokenizer


@dataclass(frozen=True)
class MovePrediction:
    move: str
    text: str
    legal: bool
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    truncated: bool


class TextGoPolicy:
    def __init__(
        self, model: Any, tokenizer: Any, *, encoding: str = "grid",
        constrained: bool = True, thinking: bool = False, max_new_tokens: int = 16,
        temperature: float = 0.0, superko: bool | None = None, num_beams: int = 1,
    ):
        if encoding not in ENCODINGS:
            raise ValueError(f"Unknown encoding: {encoding}")
        if constrained and thinking:
            raise ValueError("Move-token constraints require non-thinking generation")
        if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Invalid generation budget or temperature")
        if not isinstance(num_beams, int) or isinstance(num_beams, bool) or num_beams < 1:
            raise ValueError("num_beams must be a positive integer")
        if num_beams > 1 and (not constrained or thinking or temperature > 0):
            raise ValueError("Beam move decoding requires constrained, non-thinking, deterministic inference")
        self.model, self.tokenizer = model, tokenizer
        self.encoding, self.constrained, self.thinking = encoding, constrained, thinking
        self.max_new_tokens, self.temperature, self.superko = max_new_tokens, temperature, superko
        self.num_beams = num_beams

    def predict(self, board: BoardState, *, komi: float = 7.5, rules: str = "chinese") -> MovePrediction:
        import torch

        prompt = render_position(board, self.encoding, komi=komi, rules=rules)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.thinking,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        length = inputs.input_ids.shape[1]
        max_context = getattr(self.model.config, "max_position_embeddings", float("inf"))
        if length + self.max_new_tokens > max_context:
            raise ValueError("Position plus generation budget exceeds model context; refusing to truncate the board")
        from .rules import legality_superko

        required_superko = legality_superko(rules)
        if self.superko is not None and self.superko != required_superko:
            raise ValueError("Requested legality mask disagrees with the declared game rules")
        legal = board.legal_moves(superko=required_superko)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True, "num_beams": self.num_beams,
            "num_return_sequences": 1, "renormalize_logits": False,
        }
        if self.temperature > 0:
            kwargs.update(temperature=self.temperature, top_p=1.0, top_k=0)
        else:
            # Override checkpoint sampling defaults without ignored-flag warnings.
            kwargs.update(temperature=None, top_p=None, top_k=None)
        if self.num_beams > 1:
            from transformers import GenerationConfig

            # HF beam search computes full-vocabulary log_softmax BEFORE the
            # prefix processor masks illegal tokens. Do not renormalize that
            # mask: it changes the objective to a different, local policy.
            # Score complete strings including EOS, without a length bonus.
            # A checkpoint's optional repetition/forced-token processors would
            # also change this objective; start from neutral generation defaults.
            kwargs.update(generation_config=GenerationConfig(), use_model_defaults=False, length_penalty=0.0,
                          early_stopping="never")
        if self.constrained:
            trie = MoveTokenTrie(self.tokenizer, legal)
            if self.max_new_tokens < trie.max_tokens:
                raise ValueError(f"Need at least {trie.max_tokens} new tokens to represent every legal move and EOS")
            kwargs["prefix_allowed_tokens_fn"] = lambda _batch, ids: trie.allowed(ids[length:].tolist())
        device = self.model.device.type
        synchronize_device(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        synchronize_device(device)
        seconds = time.perf_counter() - started
        ids = generated[0, length:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        final, _, malformed = extract_final_answer(text, "qwen3" if self.thinking else "plain")
        move = "" if malformed else parse_move_answer(final)
        return MovePrediction(
            move=move, text=text, legal=move in legal, prompt_tokens=length,
            completion_tokens=len(ids), seconds=seconds,
            truncated=bool(ids and ids[-1] != self.tokenizer.eos_token_id),
        )
