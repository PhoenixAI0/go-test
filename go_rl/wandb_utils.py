from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def configure_wandb(
    enabled: bool,
    run_name: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> bool:
    """
    Configure Weights & Biases (wandb) for experiment tracking.

    Args:
        enabled: Whether to enable wandb tracking.
        run_name: Name for this training run.
        project: W&B project name.
        entity: W&B entity/team name.
        mode: W&B mode ("online", "offline", "disabled").
        config: Optional config dictionary to log to wandb.
        tags: Optional list of tags for this run.
        notes: Optional notes/description for this run.

    Returns:
        True if wandb was successfully configured, False otherwise.

    Best Practices (2026):
        - Always log hyperparameters (model, learning rate, batch size, etc.)
        - Use descriptive run names with timestamps
        - Tag runs by experiment type (sft, grpo, fp8, etc.)
        - Log GPU info, quantization settings, attention backend
        - Track memory usage, throughput, and training speed metrics
    """
    if not enabled and not project and not entity and not mode:
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; disabling W&B reporting.")
        logger.info("Install with: uv pip install wandb")
        return False

    # Set environment variables for wandb configuration
    if project:
        os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    if mode:
        os.environ["WANDB_MODE"] = mode

    os.environ.setdefault("WANDB_RUN_NAME", run_name)

    # Initialize wandb with enhanced configuration
    try:
        # Get GPU info for logging
        gpu_info = get_gpu_info()

        # Merge config with GPU info
        full_config = config or {}
        if gpu_info:
            full_config["gpu"] = gpu_info

        # Initialize wandb run
        wandb.init(
            project=project or os.environ.get("WANDB_PROJECT"),
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=run_name,
            config=full_config,
            tags=tags or [],
            notes=notes,
            resume="allow",  # Allow resuming if run exists
        )

        logger.info("Weights & Biases initialized successfully for run: %s", run_name)
        if project:
            logger.info("  Project: %s", project)
        if entity:
            logger.info("  Entity: %s", entity)
        if gpu_info:
            logger.info("  GPU: %s", gpu_info.get("name", "Unknown"))

        return True

    except Exception as e:
        logger.warning("Failed to initialize wandb: %s", e)
        logger.info("Continuing without wandb tracking")
        return False


def log_training_metrics(
    step: int,
    metrics: Dict[str, Any],
    commit: bool = True,
) -> None:
    """
    Log training metrics to wandb.

    Args:
        step: Current training step.
        metrics: Dictionary of metrics to log (loss, accuracy, etc.).
        commit: Whether to commit the log immediately.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step, commit=commit)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Failed to log metrics to wandb: %s", e)


def log_system_metrics() -> None:
    """
    Log system metrics (GPU memory, utilization, etc.) to wandb.

    This should be called periodically during training to track resource usage.
    """
    try:
        import wandb
        if wandb.run is None:
            return

        gpu_metrics = get_gpu_metrics()
        if gpu_metrics:
            wandb.log(gpu_metrics, commit=False)

    except ImportError:
        pass
    except Exception as e:
        logger.debug("Failed to log system metrics: %s", e)


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Get GPU information for logging.

    Returns:
        Dictionary with GPU info (name, memory, compute capability) or None.
    """
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
    except Exception as e:
        logger.debug("Failed to get GPU info: %s", e)
        return None


def get_gpu_metrics() -> Optional[Dict[str, float]]:
    """
    Get current GPU metrics (memory usage, utilization).

    Returns:
        Dictionary with GPU metrics or None.
    """
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
    except Exception as e:
        logger.debug("Failed to get GPU metrics: %s", e)
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
    except Exception as e:
        logger.warning("Error finishing wandb run: %s", e)
