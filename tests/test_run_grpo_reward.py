from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from go_rl.config import ReasoningConfig, RewardConfig, RolloutConfig
from go_rl.katago import KataGoResult


def _load_script_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BatchOnlyKataGo:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def analyze_positions(self, boards):
        board_list = list(boards)
        self.batch_sizes.append(len(board_list))
        return [
            KataGoResult(
                winrate=0.5 + (0.01 * len(board.stones())),
                raw={"rootInfo": {"winrate": 0.5}},
            )
            for board in board_list
        ]

    def analyze_position(self, _board):
        raise AssertionError("Expected batched analyze_positions path")


PROMPT = "\n".join(
    [
        "board_size: 9",
        "to_play: B",
        "board:",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        "move:",
    ]
)


def test_run_grpo_reward_deduplicates_base_and_candidate(monkeypatch):
    module = _load_script_module("run_grpo.py", "run_grpo_script_reward_cache")
    monkeypatch.setenv("LOCAL_RANK", "0")

    katago = _BatchOnlyKataGo()
    reward_fn = module.build_reward_fn(
        katago=katago,
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        reasoning_cfg=ReasoningConfig(profile="qwen3"),
        model_name="qwen-3-8b",
        tokenizer=None,
    )

    rewards = reward_fn(
        prompts=[PROMPT, PROMPT],
        completions=["D4", "D4"],
    )

    assert len(rewards) == 2
    assert katago.batch_sizes == [1, 1]


def test_run_grpo_reward_skips_candidate_batch_for_illegal_moves(monkeypatch):
    module = _load_script_module("run_grpo.py", "run_grpo_script_reward_illegal")
    monkeypatch.setenv("LOCAL_RANK", "0")

    katago = _BatchOnlyKataGo()
    reward_fn = module.build_reward_fn(
        katago=katago,
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        reasoning_cfg=ReasoningConfig(profile="qwen3"),
        model_name="qwen-3-8b",
        tokenizer=None,
    )

    rewards = reward_fn(
        prompts=[PROMPT, PROMPT],
        completions=["Z99", "Z99"],
    )

    assert len(rewards) == 2
    assert katago.batch_sizes == [1]


def test_run_grpo_reward_reasoning_prefixed_same_as_plain(monkeypatch):
    module = _load_script_module("run_grpo.py", "run_grpo_script_reward_reasoning")
    monkeypatch.setenv("LOCAL_RANK", "0")

    katago = _BatchOnlyKataGo()
    reward_fn = module.build_reward_fn(
        katago=katago,
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        reasoning_cfg=ReasoningConfig(profile="qwen3"),
        model_name="qwen-3-8b",
        tokenizer=None,
    )

    rewards = reward_fn(
        prompts=[PROMPT, PROMPT],
        completions=["D4", "<think>analysis</think> D4"],
    )
    assert rewards[0] == rewards[1]
