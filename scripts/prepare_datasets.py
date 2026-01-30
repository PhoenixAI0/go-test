from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from go_rl.sgf_utils import iter_sgf_samples
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

    sft_path.parent.mkdir(parents=True, exist_ok=True)
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        sft_path.open("w", encoding="utf-8") as sft_handle,
        rl_path.open("w", encoding="utf-8") as rl_handle,
    ):
        for sample in iter_sgf_samples(args.sgf_dir, max_positions=args.max_positions):
            record = {
                "prompt": sample["prompt"],
                "response": sample["target"],
                "board_size": sample["board_size"],
            }
            sft_handle.write(json.dumps(record) + "\n")
            rl_handle.write(json.dumps({"prompt": sample["prompt"]}) + "\n")
            count += 1
    logger.info("Wrote %d SFT samples to %s", count, sft_path)
    logger.info("Wrote %d RL prompts to %s", count, rl_path)


if __name__ == "__main__":
    main()
