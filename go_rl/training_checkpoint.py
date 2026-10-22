"""Optimizer and RNG continuation for local, trusted experiment checkpoints."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping

import torch


def preserve_best_snapshot(directory: str | Path, best: str | Path) -> None:
    """Carry the best model so resuming cannot forget an earlier better epoch."""
    shutil.copytree(best, Path(directory) / "best_snapshot")


def restore_best_snapshot(directory: str | Path, destination: str | Path) -> bool:
    source = Path(directory) / "best_snapshot"
    if not source.is_dir():
        return False  # Older checkpoints can still resume, with segment scope.
    shutil.copytree(source, destination)
    return True


def model_fingerprints(directory: Path) -> dict[str, str]:
    if not any(directory.rglob("*.safetensors")):
        raise ValueError("Save model weights before the optimizer checkpoint")
    files = sorted(path for path in directory.rglob("*") if path.is_file()
                   and path.name not in {"training_state.pt", "training_state.pt.tmp"})
    fingerprints = {}
    for path in files:
        with path.open("rb") as handle:
            fingerprints[str(path.relative_to(directory))] = hashlib.file_digest(handle, "sha256").hexdigest()
    return fingerprints


def save_training_state(
    directory: str | Path, optimizer: torch.optim.Optimizer, scheduler: Any,
    sample_rng: random.Random, *, step: int, contract: Mapping[str, Any], device: str,
) -> None:
    """Write state only after its corresponding immutable model snapshot exists.

    The caller owns model saving. The checksum binds this optimizer/RNG state to
    exactly those weights. A fresh checkpoint directory avoids mixing snapshots.
    """
    directory = Path(directory)
    if step < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    state = {"version": 1, "step": step, "contract": dict(contract),
             "model_sha256": model_fingerprints(directory),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "sample_rng": sample_rng.getstate(), "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state(), "device_type": device}
    if device == "mps":
        state["device_rng"] = torch.mps.get_rng_state()
    elif device == "cuda":
        state["device_rng"] = torch.cuda.get_rng_state_all()
    temporary = directory / "training_state.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, directory / "training_state.pt")


def restore_training_state(
    directory: str | Path, optimizer: torch.optim.Optimizer, scheduler: Any,
    sample_rng: random.Random, *, contract: Mapping[str, Any], device: str,
) -> int:
    """Resume the same schedule/data/objective; reject silent experiment changes."""
    directory = Path(directory)
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    if state.get("version") != 1:
        raise ValueError("Unsupported training-state format")
    if state["contract"].get("trainer_version") != contract.get("trainer_version"):
        raise ValueError(
            "Resume trainer contract version changed "
            f"({state['contract'].get('trainer_version')} -> {contract.get('trainer_version')}); "
            "use init-checkpoint for a new experiment")
    if state["contract"] != dict(contract):
        differences = sorted(key for key in set(state["contract"]) | set(contract)
                             if state["contract"].get(key) != contract.get(key))
        raise ValueError(f"Resume contract changed: {', '.join(differences)}; use init-checkpoint for a new experiment")
    if state["device_type"] != device:
        raise ValueError("Exact resume requires the same device type; use init-checkpoint to migrate")
    if state["model_sha256"] != model_fingerprints(directory):
        raise ValueError("Model weights differ from the optimizer checkpoint")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sample_rng.setstate(state["sample_rng"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device == "mps":
        torch.mps.set_rng_state(state["device_rng"])
    elif device == "cuda":
        torch.cuda.set_rng_state_all(state["device_rng"])
    return int(state["step"])
