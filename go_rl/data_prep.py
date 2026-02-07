from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .sgf_utils import iter_sgf_samples

logger = logging.getLogger(__name__)


def _write_jsonl_batch(handle, records: List[dict]) -> None:
    """Write a batch of records to JSONL file efficiently.

    Args:
        handle: File handle
        records: List of dictionaries to write
    """
    if not records:
        return
    lines = [json.dumps(record) + "\n" for record in records]
    handle.write("".join(lines))


def build_sft_dataset(
    sgf_dir: str,
    output_path: str,
    max_positions: int,
    num_workers: int = 0,
    use_parallel: bool = True,
    batch_size: int = 1000,
) -> int:
    """Build SFT dataset from SGF files.

    Args:
        sgf_dir: Directory containing SGF files
        output_path: Path to output JSONL file
        max_positions: Maximum positions to collect
        num_workers: Number of parallel workers (0 = auto-detect)
        use_parallel: Whether to use parallel processing
        batch_size: Number of records to buffer before writing

    Returns:
        Number of samples written
    """
    logger.info("Loading SGF samples from %s", sgf_dir)
    logger.debug("Max positions: %d", max_positions)
    logger.debug("Parallel processing: %s (workers: %d)", use_parallel, num_workers)
    logger.debug("Batch size: %d", batch_size)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing dataset to %s", output)

    count = 0
    batch: List[dict] = []

    iter_kwargs = {}
    if num_workers != 0:
        iter_kwargs["num_workers"] = num_workers
    if use_parallel is not True:
        iter_kwargs["use_parallel"] = use_parallel

    with output.open("w", encoding="utf-8", buffering=8192 * 16) as handle:
        for sample in iter_sgf_samples(
            sgf_dir,
            max_positions=max_positions,
            **iter_kwargs,
        ):
            record = {
                "prompt": sample["prompt"],
                "response": sample["target"],
                "board_size": sample["board_size"],
            }
            batch.append(record)
            count += 1

            # Write batch when full
            if len(batch) >= batch_size:
                _write_jsonl_batch(handle, batch)
                batch = []

                if count % 10000 == 0:
                    logger.info("Written %d samples...", count)

        # Write remaining batch
        if batch:
            _write_jsonl_batch(handle, batch)

    logger.info("Finished writing dataset to %s", output)
    return count
