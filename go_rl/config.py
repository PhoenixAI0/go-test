from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _expand_path(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _validate_path(value: str, label: str) -> None:
    if value.startswith("/path/to") or "$" in value:
        raise ValueError(f"{label} must be set to a real path.")


@dataclass
class KataGoConfig:
    binary_path: str
    model_path: str
    config_path: str
    board_size: int = 9
    rules: str = "chinese"
    komi: float = 7.5

    def __post_init__(self) -> None:
        self.binary_path = _expand_path(self.binary_path)
        self.model_path = _expand_path(self.model_path)
        self.config_path = _expand_path(self.config_path)
        _validate_path(self.binary_path, "KataGo binary_path")
        _validate_path(self.model_path, "KataGo model_path")
        _validate_path(self.config_path, "KataGo config_path")


@dataclass
class RolloutConfig:
    num_samples: int = 8
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 4
    stop_at_pass: bool = True


@dataclass
class RewardConfig:
    winrate_scale: float = 1.0
    legal_bonus: float = 0.2
    illegal_penalty: float = -1.0
    length_penalty: float = -0.05
    full_game_bonus: float = 0.5
    strict_legality: bool = False


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: str
    log_dir: str = "logs/grpo"
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 1e-5
    num_steps: int = 2000
    seed: int = 3407
    lora_rank: Optional[int] = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_lora: bool = True
    grpo_mini_batch: Optional[int] = 1
    grpo_logit_chunk_multiplier: Optional[int] = 4
    use_8bit_optimizer: bool = True
    quantization_bits: int = 8

    def __post_init__(self) -> None:
        self.output_dir = _expand_path(self.output_dir)
        self.log_dir = _expand_path(self.log_dir)
        if self.quantization_bits not in (0, 4, 8, 16, 32):
            raise ValueError(
                "quantization_bits must be one of 0, 4, 8, 16, or 32."
            )


@dataclass
class SftConfig:
    dataset_path: str
    model_name: str
    output_dir: str
    log_dir: str = "logs/sft"
    max_seq_length: int = 1024
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_packing: bool = True
    use_8bit_optimizer: bool = True
    quantization_bits: int = 8
    seed: int = 3407

    def __post_init__(self) -> None:
        self.dataset_path = _expand_path(self.dataset_path)
        self.output_dir = _expand_path(self.output_dir)
        self.log_dir = _expand_path(self.log_dir)
        if self.quantization_bits not in (0, 4, 8, 16, 32):
            raise ValueError(
                "quantization_bits must be one of 0, 4, 8, 16, or 32."
            )


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load config files.") from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def load_config(path: str, cls: type) -> Any:
    data = _load_yaml(path)
    return cls(**data)
