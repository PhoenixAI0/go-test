from __future__ import annotations

import math
import os
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Optional, TypeVar

_ConfigT = TypeVar("_ConfigT")

VALID_QUANTIZATION_BITS = (0, 4, 8, 16, 32)


def _expand_path(value: str, base_dir: Optional[str] = None) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if base_dir and not os.path.isabs(expanded):
        return os.path.abspath(os.path.join(base_dir, expanded))
    return expanded


def _validate_path(value: str, label: str) -> None:
    if value.startswith("/path/to") or "$" in value:
        raise ValueError(f"{label} must be set to a real path.")


@dataclass
class KataGoConfig:
    binary_path: str
    model_path: str
    config_path: str
    config_base_dir: InitVar[Optional[str]] = None
    board_size: int = 9
    rules: str = "chinese"
    komi: float = 7.5
    katago_cuda_visible_devices: Optional[str] = None
    request_timeout_seconds: Optional[float] = 30.0
    katago_pool_size: int = 1

    def __post_init__(self, config_base_dir: Optional[str]) -> None:
        self.binary_path = _expand_path(self.binary_path, base_dir=config_base_dir)
        self.model_path = _expand_path(self.model_path, base_dir=config_base_dir)
        self.config_path = _expand_path(self.config_path, base_dir=config_base_dir)
        _validate_path(self.binary_path, "KataGo binary_path")
        _validate_path(self.model_path, "KataGo model_path")
        _validate_path(self.config_path, "KataGo config_path")
        if self.request_timeout_seconds is not None and self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0 when provided.")
        if self.katago_pool_size < 1:
            raise ValueError("katago_pool_size must be >= 1.")


@dataclass
class RolloutConfig:
    num_samples: int = 8
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 4
    # Disable top-k truncation so sampling matches the temperature-softmax
    # log probabilities used by the policy-gradient objective.
    top_k: int = 0

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1.")


@dataclass
class GRPOObjectiveConfig:
    """Explicit, reproducible TRL objective settings (top-level ``grpo`` YAML).

    Dr. GRPO removes per-response length and per-group variance weighting.
    These are baseline choices to ablate, not a claim of optimal Go strength.
    """

    loss_type: str = "dr_grpo"
    scale_rewards: str = "none"
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    num_iterations: int = 1
    importance_sampling_level: str = "token"
    mask_truncated_completions: bool = False
    top_entropy_quantile: float = 1.0

    def __post_init__(self) -> None:
        if self.loss_type not in {"grpo", "dapo", "bnpo", "dr_grpo"}:
            raise ValueError("loss_type must be grpo, dapo, bnpo, or dr_grpo.")
        if self.scale_rewards not in {"group", "batch", "none"}:
            raise ValueError("scale_rewards must be group, batch, or none.")
        if self.importance_sampling_level not in {"token", "sequence"}:
            raise ValueError("importance_sampling_level must be token or sequence.")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative.")
        if not math.isfinite(self.epsilon) or not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must be finite and in [0, 1).")
        if not math.isfinite(self.epsilon_high) or self.epsilon_high < 0:
            raise ValueError("epsilon_high must be finite and nonnegative.")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1.")
        if not 0 < self.top_entropy_quantile <= 1:
            raise ValueError("top_entropy_quantile must be in (0, 1].")


@dataclass
class FullGameConfig:
    enabled: bool = True
    max_plies: int = 300
    termination_two_passes: bool = True
    opponent_policy: str = "katago_best"
    model_color_mode: str = "alternate"
    seed_empty_board_prob: float = 0.4
    max_model_turns: int = 150

    def __post_init__(self) -> None:
        if not 0.0 <= self.seed_empty_board_prob <= 1.0:
            raise ValueError("seed_empty_board_prob must be in [0.0, 1.0].")
        if self.max_plies < 1:
            raise ValueError("max_plies must be >= 1.")
        if self.max_model_turns < 1:
            raise ValueError("max_model_turns must be >= 1.")
        if self.opponent_policy not in {"katago_best"}:
            raise ValueError("opponent_policy must be 'katago_best'.")
        if self.model_color_mode not in {"alternate", "black", "white"}:
            raise ValueError("model_color_mode must be alternate, black, or white.")


@dataclass
class RewardConfig:
    winrate_scale: float = 1.0
    legal_bonus: float = 0.2
    illegal_penalty: float = -1.0
    full_game_bonus: float = 0.5
    scorelead_scale: float = 0.02
    terminal_outcome_bonus: float = 1.0
    illegal_move_ends_game: bool = True
    strict_legality: bool = False
    # Legality checking: basic (occupancy), ko (simple ko), or superko (positional repeat).
    legality_level: Optional[str] = None


@dataclass
class ReasoningConfig:
    enabled: bool = True
    profile: str = "auto"
    # A sampled reasoning prefix is part of the policy action and needs credit
    # from the same outcome reward. Final-only optimization is an ablation.
    optimize_final_only: bool = False
    malformed_policy: str = "mask_all"
    qwen_enable_thinking: bool = True
    gpt_oss_reasoning_effort: str = "medium"
    glm_thinking_type: str = "enabled"
    glm_clear_thinking: bool = False
    future_reasoning_length_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.profile not in {"auto", "qwen3", "gpt_oss", "glm47", "plain"}:
            raise ValueError("profile must be one of auto, qwen3, gpt_oss, glm47, plain.")
        if self.malformed_policy not in {"mask_all"}:
            raise ValueError("malformed_policy must be 'mask_all'.")
        if self.gpt_oss_reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("gpt_oss_reasoning_effort must be low, medium, or high.")
        if self.glm_thinking_type not in {"enabled", "disabled"}:
            raise ValueError("glm_thinking_type must be enabled or disabled.")
        if self.future_reasoning_length_penalty != 0.0:
            raise ValueError("future_reasoning_length_penalty must remain 0.0 for now.")


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: str
    input_encoding: str = "compact"
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
    use_8bit_optimizer: bool = True
    quantization_bits: int = 8
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)

    def __post_init__(self) -> None:
        if self.input_encoding not in {"legacy", "grid", "compact", "stones"}:
            raise ValueError("input_encoding must be legacy, grid, compact, or stones.")
        self.learning_rate = float(self.learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        self.output_dir = _expand_path(self.output_dir)
        self.log_dir = _expand_path(self.log_dir)
        if isinstance(self.reasoning, dict):
            self.reasoning = ReasoningConfig(**self.reasoning)
        if self.quantization_bits not in VALID_QUANTIZATION_BITS:
            raise ValueError(
                f"quantization_bits must be one of {', '.join(map(str, VALID_QUANTIZATION_BITS))}."
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
    eval_split_ratio: float = 0.05
    eval_steps: int = 200
    input_encoding: str = "legacy"
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)

    def __post_init__(self) -> None:
        self.dataset_path = _expand_path(self.dataset_path)
        self.learning_rate = float(self.learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        self.output_dir = _expand_path(self.output_dir)
        self.log_dir = _expand_path(self.log_dir)
        if isinstance(self.reasoning, dict):
            self.reasoning = ReasoningConfig(**self.reasoning)
        if self.quantization_bits not in VALID_QUANTIZATION_BITS:
            raise ValueError(
                f"quantization_bits must be one of {', '.join(map(str, VALID_QUANTIZATION_BITS))}."
            )
        if self.eval_split_ratio < 0.0 or self.eval_split_ratio >= 1.0:
            raise ValueError("eval_split_ratio must be in [0.0, 1.0).")
        if self.eval_steps < 1:
            raise ValueError("eval_steps must be >= 1.")
        if self.input_encoding not in {"legacy", "grid", "compact", "stones"}:
            raise ValueError("input_encoding must be legacy, grid, compact, or stones.")


def load_yaml_mapping(
    path: str,
    missing_dependency_message: str = "PyYAML is required to load config files.",
) -> dict[str, object]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(missing_dependency_message) from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def resolve_config_base_dir(path: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve().parent)


def run_name_for_config(prefix: str, config_path: str) -> str:
    return f"{prefix}_{Path(config_path).stem}"


def load_config(path: str, cls: type[_ConfigT]) -> _ConfigT:
    data = load_yaml_mapping(path)
    if cls is KataGoConfig:
        return cls(config_base_dir=resolve_config_base_dir(path), **data)
    return cls(**data)


_load_yaml = load_yaml_mapping
