from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


def quantization_kwargs(bits: int) -> Dict[str, bool]:
    if bits == 4:
        return {"load_in_4bit": True}
    if bits == 8:
        return {"load_in_8bit": True}
    if bits in (0, 16, 32):
        return {}
    raise ValueError(
        "quantization_bits must be one of 0, 4, 8, 16, or 32."
    )


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2 or 1))


def configure_torch_for_speed() -> None:
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    logger.info("Enabled TF32 and cuDNN benchmarking for faster training.")
