from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from go_rl.sgf_utils import iter_sgf_samples
from go_rl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _write_jsonl_batch(handle, records: List[dict]) -> None:
    """Write a batch of records to JSONL file efficiently."""
    if not records:
        return
    lines = [json.dumps(record) + "\n" for record in records]
    handle.write("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True, help="Directory containing SGF files")
    parser.add_argument("--output-dir", default="data", help="Output directory for datasets")
    parser.add_argument("--tag", default="9x9", help="Tag for output filenames")
    parser.add_argument("--max-positions", type=int, default=200000, help="Maximum positions to collect")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect, 1 = sequential)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
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
    logger.debug("Parallel processing: %s (workers: %d)", not args.no_parallel, args.num_workers)
    logger.debug("Batch size: %d", args.batch_size)

    output_root = Path(args.output_dir)
    sft_path = output_root / f"sft_{args.tag}.jsonl"
    rl_path = output_root / f"rl_prompts_{args.tag}.jsonl"

    sft_path.parent.mkdir(parents=True, exist_ok=True)
    rl_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    sft_batch: List[dict] = []
    rl_batch: List[dict] = []

    with (
        sft_path.open("w", encoding="utf-8", buffering=8192 * 16) as sft_handle,
        rl_path.open("w", encoding="utf-8", buffering=8192 * 16) as rl_handle,
    ):
        for sample in iter_sgf_samples(
            args.sgf_dir,
            max_positions=args.max_positions,
            num_workers=args.num_workers,
            use_parallel=not args.no_parallel,
        ):
            sft_record = {
                "prompt": sample["prompt"],
                "response": sample["target"],
                "board_size": sample["board_size"],
            }
            rl_record = {"prompt": sample["prompt"]}

            sft_batch.append(sft_record)
            rl_batch.append(rl_record)
            count += 1

            # Write batches when full
            if len(sft_batch) >= args.batch_size:
                _write_jsonl_batch(sft_handle, sft_batch)
                _write_jsonl_batch(rl_handle, rl_batch)
                sft_batch = []
                rl_batch = []

                if count % 10000 == 0:
                    logger.info("Written %d samples...", count)

        # Write remaining batches
        if sft_batch:
            _write_jsonl_batch(sft_handle, sft_batch)
            _write_jsonl_batch(rl_handle, rl_batch)

    logger.info("Wrote %d SFT samples to %s", count, sft_path)
    logger.info("Wrote %d RL prompts to %s", count, rl_path)


if __name__ == "__main__":
    main()
