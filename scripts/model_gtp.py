#!/usr/bin/env python3
"""Serve a text Go model or categorical policy checkpoint through GTP.

Example:
  uv run python scripts/model_gtp.py --model Qwen/Qwen3-0.6B --encoding compact
  uv run python scripts/model_gtp.py --checkpoint outputs/policy_19x19

The match referee owns clocks and terminal scoring. This engine deliberately
has no final_score command; it never presents model estimates as game results.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

from go_rl.gtp_engine import CategoricalMoveChooser, ModelGtpEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default=None, help="Hugging Face model/PEFT adapter (default Qwen/Qwen3-0.6B)")
    source.add_argument("--checkpoint", help="Go policy/value checkpoint directory containing go_policy.json")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--board-size", type=int)
    parser.add_argument("--encoding", choices=["legacy", "grid", "compact", "stones", "cells"])
    parser.add_argument("--rules", help="Exact KataGo rules; defaults to checkpoint metadata or chinese")
    parser.add_argument("--komi", type=float)
    parser.add_argument("--name", default="GoRL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Text moves only: approximate legal complete-string maximum likelihood; 1=greedy")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--thinking", action="store_true")
    constraints = parser.add_mutually_exclusive_group()
    constraints.add_argument("--constrained", action="store_true", dest="constrained", default=True)
    constraints.add_argument("--unconstrained", action="store_false", dest="constrained")
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            torch.manual_seed(args.seed)
            metadata = {}
            if args.checkpoint:
                if args.num_beams != 1:
                    raise ValueError("--num-beams is available only for text-model moves")
                if args.thinking:
                    raise ValueError("Categorical policy checkpoints do not generate thought tokens")
                from go_rl.policy_model import load_policy_model
                model, tokenizer, metadata = load_policy_model(args.checkpoint, device=args.device)
                encoding = args.encoding or metadata.get("encoding", "compact")
                chooser = CategoricalMoveChooser(model, tokenizer, encoding=encoding,
                                                 temperature=args.temperature, constrained=args.constrained)
                supported_sizes = {model.board_size}
            else:
                from go_rl.model_interface import TextGoPolicy, load_text_model
                model_path = args.model or "Qwen/Qwen3-0.6B"
                metadata_path = Path(model_path) / "go_text.json"
                metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
                encoding = args.encoding or metadata.get("encoding") or "compact"
                if encoding == "cells":
                    raise ValueError("cells encoding is available only for categorical policy checkpoints")
                model, tokenizer = load_text_model(model_path, args.device, args.revision)
                policy = TextGoPolicy(model, tokenizer, encoding=encoding, constrained=args.constrained,
                                      thinking=args.thinking, max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature, num_beams=args.num_beams)
                def chooser(board, **kwargs):
                    return policy.predict(board, **kwargs).move
                supported_sizes = range(2, 20)
            engine = ModelGtpEngine(
                chooser, board_size=args.board_size if args.board_size is not None else metadata.get("board_size", 19),
                komi=args.komi if args.komi is not None else metadata.get("komi", 7.5),
                rules=args.rules or metadata.get("rules", "chinese"), name=args.name,
                supported_sizes=supported_sizes,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Could not start model GTP engine: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    engine.run()


if __name__ == "__main__":
    main()
