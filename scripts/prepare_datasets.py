from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from go_rl.data_prep import run_go_parser, write_jsonl_batch
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True, help="Directory containing SGF files")
    parser.add_argument("--output-dir", default="data", help="Output directory for datasets")
    parser.add_argument("--tag", default="9x9", help="Tag for output filenames")
    parser.add_argument("--max-positions", type=int, default=200000, help="Maximum positions to collect")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Go parser worker count (0 = auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for buffered writes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Preparing SFT and RL datasets")
    logger.debug("Batch size: %d", args.batch_size)

    output_root = Path(args.output_dir)
    sft_path = output_root / f"sft_{args.tag}.jsonl"
    rl_path = output_root / f"rl_prompts_{args.tag}.jsonl"

    sft_path.parent.mkdir(parents=True, exist_ok=True)
    rl_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    sft_batch: List[dict] = []
    rl_batch: List[dict] = []

    go_parser_output = output_root / f"__go_parser_{args.tag}.jsonl"
    logger.info("Building intermediate dataset with Go parser")
    run_go_parser(
        sgf_dir=args.sgf_dir,
        output_path=str(go_parser_output),
        max_positions=args.max_positions,
        num_workers=args.num_workers,
    )

    with (
        sft_path.open("w", encoding="utf-8", buffering=8192 * 16) as sft_handle,
        rl_path.open("w", encoding="utf-8", buffering=8192 * 16) as rl_handle,
    ):
        logger.info("Processing Go parser output: %s", go_parser_output)
        with go_parser_output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                sft_record = {
                    "prompt": sample["prompt"],
                    "response": sample["response"],
                    "board_size": sample.get("board_size"),
                }
                rl_record = {"prompt": sample["prompt"]}

                sft_batch.append(sft_record)
                rl_batch.append(rl_record)
                count += 1

                if len(sft_batch) >= args.batch_size:
                    write_jsonl_batch(sft_handle, sft_batch)
                    write_jsonl_batch(rl_handle, rl_batch)
                    sft_batch = []
                    rl_batch = []

                    if count % 10000 == 0:
                        logger.info("Written %d samples...", count)

        if sft_batch:
            write_jsonl_batch(sft_handle, sft_batch)
            write_jsonl_batch(rl_handle, rl_batch)

    if go_parser_output.exists():
        go_parser_output.unlink()

    logger.info("Wrote %d SFT samples to %s", count, sft_path)
    logger.info("Wrote %d RL prompts to %s", count, rl_path)


if __name__ == "__main__":
    main()
