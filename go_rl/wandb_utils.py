from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Optional

logger = logging.getLogger(__name__)


def configure_wandb(
    enabled: bool,
    run_name: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
    config: Mapping[str, object] | None = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> bool:
    if not enabled and not project and not entity and not mode:
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; disabling W&B reporting.")
        logger.info("Sync project dependencies with: uv sync --locked")
        return False

    if project:
        os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    if mode:
        os.environ["WANDB_MODE"] = mode

    os.environ.setdefault("WANDB_RUN_NAME", run_name)

    try:
        gpu_info = get_gpu_info()
        full_config = dict(config or {})
        if gpu_info:
            full_config["gpu"] = gpu_info

        wandb.init(
            project=project or os.environ.get("WANDB_PROJECT"),
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=run_name,
            config=full_config,
            tags=tags or [],
            notes=notes,
            resume="allow",
        )

        logger.info("Weights & Biases initialized successfully for run: %s", run_name)
        if project:
            logger.info("  Project: %s", project)
        if entity:
            logger.info("  Entity: %s", entity)
        if gpu_info:
            logger.info("  GPU: %s", gpu_info.get("name", "Unknown"))

        return True

    except Exception as exc:
        logger.warning("Failed to initialize wandb: %s", exc)
        logger.info("Continuing without wandb tracking")
        return False


def log_training_metrics(step: int, metrics: Mapping[str, object]) -> None:
    """Log one row against the trainer's step axis.

    Post-training logs can't use wandb's own step: it only increases, so earlier
    steps would be dropped. Transformers charts every metric by train/global_step.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({**metrics, "train/global_step": step})
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Failed to log metrics to wandb: %s", exc)


def log_prefixed_history(
    log_history: list[dict[str, object]],
    wandb_enabled: bool,
    metric_prefix: str,
) -> None:
    if not wandb_enabled:
        return
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        metrics: dict[str, float] = {}
        for key, value in entry.items():
            if key == "step":
                continue
            if "reward" in key.lower():
                metrics[f"{metric_prefix}{key}"] = float(value)
        if "loss" in entry:
            metrics[f"{metric_prefix}loss"] = float(entry["loss"])
        if metrics:
            log_training_metrics(step=int(step), metrics=metrics)


def log_system_metrics() -> None:
    try:
        import wandb

        if wandb.run is None:
            return

        gpu_metrics = get_gpu_metrics()
        if gpu_metrics:
            wandb.log(gpu_metrics, commit=False)

    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Failed to log system metrics: %s", exc)


def get_gpu_info() -> dict[str, object] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "available": True,
            "name": torch.cuda.get_device_name(device),
            "count": torch.cuda.device_count(),
            "memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        logger.debug("Failed to get GPU info: %s", exc)
        return None


def get_gpu_metrics() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)

        return {
            "gpu/memory_allocated_gb": allocated,
            "gpu/memory_reserved_gb": reserved,
            "gpu/memory_max_allocated_gb": max_allocated,
        }
    except Exception as exc:
        logger.debug("Failed to get GPU metrics: %s", exc)
        return None


def finish_wandb() -> None:
    """Finish the wandb run and upload final results."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            logger.info("Weights & Biases run finished successfully")
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Error finishing wandb run: %s", exc)
