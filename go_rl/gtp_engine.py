"""A small GTP front end for model policies, with the rules engine as referee.

No scoring estimates are exposed as final scores. Use an independent KataGo
scorer in ``gtp_match``. Generation errors never silently become pass moves.
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
import traceback
from collections.abc import Callable, Collection
from typing import TextIO

from .board import BoardState, coords_to_gtp, gtp_to_coords
from .rules import legality_superko, normalize_rules

MoveChooser = Callable[..., str]


class ModelGtpEngine:
    COMMANDS = (
        "protocol_version", "name", "version", "known_command", "list_commands",
        "boardsize", "clear_board", "komi", "play", "genmove", "undo", "showboard",
        "kata-set-rules", "kata-get-rules", "quit",
    )

    def __init__(
        self, choose_move: MoveChooser, *, board_size: int = 19,
        komi: float = 7.5, rules: str = "chinese", name: str = "GoRL",
        supported_sizes: Collection[int] = range(2, 20),
    ) -> None:
        if board_size not in supported_sizes:
            raise ValueError(f"Policy does not support board size {board_size}")
        self.choose_move = choose_move
        self.supported_sizes = frozenset(supported_sizes)
        self.board = BoardState.empty(board_size)
        self.komi = self._parse_komi(str(komi))
        self.rules = rules
        self.superko = legality_superko(rules)
        self.name = name
        self.stopped = False

    @staticmethod
    def _parse_komi(value: str) -> float:
        komi = float(value)
        if not math.isfinite(komi) or not -400 <= komi <= 400 or not (2 * komi).is_integer():
            raise ValueError("komi must be a finite integer or half-integer in [-400, 400]")
        return komi

    @staticmethod
    def _color(value: str) -> str:
        colors = {"b": "B", "black": "B", "w": "W", "white": "W"}
        if value.lower() not in colors:
            raise ValueError("color must be black or white")
        return colors[value.lower()]

    @staticmethod
    def _arity(args: list[str], count: int) -> None:
        if len(args) != count:
            raise ValueError(f"expected {count} argument(s)")

    def _apply(self, color: str, move: str) -> str:
        coords = gtp_to_coords(move, self.board.size)
        canonical = coords_to_gtp(coords, self.board.size)
        if canonical not in self.board.legal_moves(color, superko=self.superko):
            raise ValueError(f"illegal move: {color} {canonical}")
        self.board.apply_move(color, coords, strict=True)
        return canonical

    def execute(self, command: str, args: list[str]) -> str:
        if command not in self.COMMANDS:
            raise ValueError("unknown command")
        if command == "protocol_version":
            self._arity(args, 0)
            return "2"
        if command == "name":
            self._arity(args, 0)
            return self.name
        if command == "version":
            self._arity(args, 0)
            return "0.2"
        if command == "known_command":
            self._arity(args, 1)
            return "true" if args[0].lower() in self.COMMANDS else "false"
        if command == "list_commands":
            self._arity(args, 0)
            return "\n".join(self.COMMANDS)
        if command == "boardsize":
            self._arity(args, 1)
            size = int(args[0])
            if size not in self.supported_sizes:
                raise ValueError(f"unacceptable size: {size}")
            self.board = BoardState.empty(size)
        elif command == "clear_board":
            self._arity(args, 0)
            self.board = BoardState.empty(self.board.size)
        elif command == "komi":
            self._arity(args, 1)
            self.komi = self._parse_komi(args[0])
        elif command == "kata-set-rules":
            if not args:
                raise ValueError("expected a rules name or JSON object")
            rules = " ".join(args)
            superko = legality_superko(rules)
            # Validate before changing either field. Existing history is retained.
            self.rules, self.superko = rules, superko
        elif command == "kata-get-rules":
            self._arity(args, 0)
            return json.dumps(normalize_rules(self.rules), sort_keys=True)
        elif command == "play":
            self._arity(args, 2)
            self._apply(self._color(args[0]), args[1])
        elif command == "genmove":
            self._arity(args, 1)
            color = self._color(args[0])
            position = self.board.clone()
            if color != position.next_player():
                if position.moves:
                    raise ValueError("genmove color differs from player to move; use play to supply the missing turn")
                position.next_player_override = color
            move = self.choose_move(position, komi=self.komi, rules=self.rules)
            if not isinstance(move, str) or not move.strip():
                raise ValueError("model did not produce a move")
            if move.strip().upper() == "RESIGN":
                return "resign"
            return self._apply(color, move).lower()
        elif command == "undo":
            self._arity(args, 0)
            if not self.board.moves:
                raise ValueError("cannot undo")
            self.board.moves.pop()
        elif command == "showboard":
            self._arity(args, 0)
            symbols = ".XO"
            lines = [f"{self.board.size - row:2d} " + " ".join(symbols[cell] for cell in cells)
                     for row, cells in enumerate(self.board.current_board())]
            columns = "ABCDEFGHJKLMNOPQRST"[:self.board.size]
            lines.append("   " + " ".join(columns))
            return "\n".join(lines)
        elif command == "quit":
            self._arity(args, 0)
            self.stopped = True
        return ""

    def run(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
            stderr: TextIO = sys.stderr) -> None:
        """Read GTP until quit/EOF; only protocol responses go to stdout."""
        for line in stdin:
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            parts = clean.split()
            request_id = parts.pop(0) if parts[0].isascii() and parts[0].isdigit() else ""
            try:
                if not parts:
                    raise ValueError("missing command")
                command, args = parts[0].lower(), parts[1:]
                # Model libraries sometimes print progress to stdout. Keep the
                # GTP transport clean during both generation and error handling.
                with contextlib.redirect_stdout(stderr):
                    text = self.execute(command, args)
                response = f"={request_id}" + (f" {text}" if text else "")
            except ValueError as exc:
                response = f"?{request_id} " + " ".join(str(exc).split())
            except Exception:
                traceback.print_exc(file=stderr)
                response = f"?{request_id} model error; see stderr"
            stdout.write(response + "\n\n")
            stdout.flush()
            if self.stopped:
                break


class CategoricalMoveChooser:
    """Inference adapter for a saved finite-action policy/value checkpoint."""

    def __init__(self, model, tokenizer, *, encoding: str = "compact",
                 temperature: float = 0.0, constrained: bool = True):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if getattr(model, "readout", "last") in {"cells", "cell_attention"} and encoding != "cells":
            raise ValueError("Spatial policy readout requires the cells encoding")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.encoding = encoding
        self.temperature = temperature
        self.constrained = constrained

    def __call__(self, board: BoardState, *, komi: float, rules: str) -> str:
        import torch
        from .model_interface import action_to_move, legal_action_mask
        from .policy_model import collate_policy_inputs, encode_policy_position

        if board.size != self.model.board_size:
            raise ValueError("Checkpoint board size differs from the game")
        superko = legality_superko(rules)
        ids = encode_policy_position(board, self.tokenizer, self.encoding, komi=komi, rules=rules)
        max_context = getattr(self.model.backbone.config, "max_position_embeddings", float("inf"))
        if len(ids) > max_context:
            raise ValueError("Position exceeds model context; refusing to truncate the board")
        inputs = collate_policy_inputs([ids], self.tokenizer, self.model.device)
        legal = torch.tensor([legal_action_mask(board, superko=superko)],
                             dtype=torch.bool, device=self.model.device) if self.constrained else None
        with torch.inference_mode():
            logits = self.model(**inputs, legal_mask=legal).logits[0]
            if self.temperature:
                action = torch.multinomial((logits / self.temperature).softmax(-1), 1).item()
            else:
                action = logits.argmax().item()
        return action_to_move(action, board.size)
