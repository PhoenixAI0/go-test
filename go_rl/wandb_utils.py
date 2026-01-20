from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def configure_wandb(
    enabled: bool,
    run_name: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
) -> bool:
    if not enabled and not project and not entity and not mode:
        return False

    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb not installed; disabling W&B reporting.")
        return False

    if project:
        os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    if mode:
        os.environ["WANDB_MODE"] = mode

    os.environ.setdefault("WANDB_RUN_NAME", run_name)
    logger.info("Enabled Weights & Biases reporting for %s", run_name)
    return True
