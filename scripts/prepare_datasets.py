from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from go_rl.sgf_utils import load_sgf_dir
from go_rl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--tag", default="9x9")
    parser.add_argument("--max-positions", type=int, default=200000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Preparing SFT and RL datasets")
    output_root = Path(args.output_dir)
    sft_path = output_root / f"sft_{args.tag}.jsonl"
    rl_path = output_root / f"rl_prompts_{args.tag}.jsonl"

    samples = load_sgf_dir(args.sgf_dir, max_positions=args.max_positions)
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with sft_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "prompt": sample["prompt"],
                "response": sample["target"],
                "board_size": sample["board_size"],
            }
            handle.write(json.dumps(record) + "\n")
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    with rl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({"prompt": sample["prompt"]}) + "\n")
    logger.info("Wrote %d SFT samples to %s", len(samples), sft_path)
    logger.info("Wrote %d RL prompts to %s", len(samples), rl_path)


if __name__ == "__main__":
    main()
