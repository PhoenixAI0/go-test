from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from collections.abc import Iterable
from typing import Optional

from .shared_types import EnvironmentInfo, GpuInfo, RunLogPayload

logger = logging.getLogger(__name__)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def _gpu_info() -> GpuInfo:
    try:
        import torch
    except ImportError:
        return {"available": False, "detail": "torch not installed"}

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "backend": "cuda",
                "device_count": torch.cuda.device_count(),
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "torch_cuda": torch.version.cuda,
            }
    except (AttributeError, RuntimeError):
        return {
            "available": False,
            "detail": "cuda/mps not available",
            "torch_cuda": torch.version.cuda,
        }

    mps_info = {}
    if hasattr(torch.backends, "mps"):
        try:
            mps_info = {
                "torch_mps_built": torch.backends.mps.is_built(),
                "torch_mps_available": torch.backends.mps.is_available(),
            }
        except (AttributeError, RuntimeError):
            mps_info = {
                "torch_mps_built": False,
                "torch_mps_available": False,
            }

    if mps_info.get("torch_mps_available"):
        return {
            "available": True,
            "backend": "mps",
            "name": "Apple MPS",
            "torch_cuda": torch.version.cuda,
            **mps_info,
        }

    return {
        "available": False,
        "detail": "cuda/mps not available",
        "torch_cuda": torch.version.cuda,
        **mps_info,
    }


def collect_env_info(packages: Iterable[str]) -> EnvironmentInfo:
    info: EnvironmentInfo = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "packages": {name: _package_version(name) for name in packages},
        "gpu": _gpu_info(),
    }
    logger.debug("Collected environment info for %d packages", len(info["packages"]))
    return info


def write_run_log(
    log_dir: str,
    run_name: str,
    config_path: str,
    config_snapshot: dict[str, object],
    extra: Optional[dict[str, object]] = None,
    packages: Optional[Iterable[str]] = None,
) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    payload: RunLogPayload = {
        "run_name": run_name,
        "timestamp_utc": timestamp,
        "config_path": config_path,
        "config": config_snapshot,
        "environment": collect_env_info(packages or []),
    }
    if extra:
        payload["extra"] = extra
    log_path = log_root / f"{run_name}_{timestamp}.json"
    logger.info("Writing run log to %s", log_path)
    log_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return str(log_path)
