from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from go_rl.config import FullGameConfig, RewardConfig, RolloutConfig
from go_rl.fullgame_rollout import FullGameEpisodeResult


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


def _prompt(board_size: int = 19) -> str:
    rows = ["." * board_size for _ in range(board_size)]
    return "\n".join(
        [
            f"board_size: {board_size}",
            "to_play: B",
            "board:",
            *rows,
            "move:",
        ]
    )


class _NoopKataGo:
    pass


def test_build_fullgame_reward_fn_returns_scalar_per_sample(monkeypatch):
    module = _load_script_module("run_grpo_fullgame.py", "run_grpo_fullgame_script")

    def fake_simulate(**_kwargs):
        return FullGameEpisodeResult(
            total_reward=1.25,
            scorelead_reward=0.5,
            terminal_reward=1.0,
            legality_reward_total=-0.2,
            length_reward_total=-0.05,
            model_turns=2,
            plies_played=3,
            truncated=False,
            terminated_by_illegal=False,
            pass_terminated=False,
            model_color="B",
            scorelead_delta_sum=25.0,
            final_outcome=1.0,
            katago_fallback_passes=0,
        )

    monkeypatch.setattr(module, "simulate_full_game", fake_simulate)
    monkeypatch.setenv("LOCAL_RANK", "0")

    reward_fn = module.build_fullgame_reward_fn(
        model=object(),
        tokenizer=object(),
        katago=_NoopKataGo(),
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        full_cfg=FullGameConfig(),
        seed=123,
    )

    rewards = reward_fn(
        prompts=[_prompt(), _prompt()],
        completions=["D4", "Q16"],
    )
    assert rewards == [1.25, 1.25]


def test_fullgame_reward_fn_runs_on_worker_rank(monkeypatch):
    module = _load_script_module("run_grpo_fullgame.py", "run_grpo_fullgame_script_rank")
    monkeypatch.setenv("LOCAL_RANK", "1")

    reward_fn = module.build_fullgame_reward_fn(
        model=object(),
        tokenizer=object(),
        katago=_NoopKataGo(),
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        full_cfg=FullGameConfig(),
        seed=123,
    )

    from types import SimpleNamespace
    monkeypatch.setattr(module, "simulate_full_game", lambda **kwargs: SimpleNamespace(
        total_reward=0.5, truncated=False, terminated_by_illegal=False, katago_fallback_passes=0))
    assert reward_fn(prompts=[_prompt()], completions=["D4"]) == [0.5]


def test_fullgame_reward_fn_uses_configured_max_plies(monkeypatch):
    module = _load_script_module("run_grpo_fullgame.py", "run_grpo_fullgame_script_maxplies")
    monkeypatch.setenv("LOCAL_RANK", "0")

    observed = {"max_plies": None}

    def fake_simulate(**kwargs):
        observed["max_plies"] = kwargs["fullgame_cfg"].max_plies
        return FullGameEpisodeResult(
            total_reward=0.0,
            scorelead_reward=0.0,
            terminal_reward=0.0,
            legality_reward_total=0.0,
            length_reward_total=0.0,
            model_turns=1,
            plies_played=1,
            truncated=False,
            terminated_by_illegal=False,
            pass_terminated=False,
            model_color="B",
            scorelead_delta_sum=0.0,
            final_outcome=0.0,
            katago_fallback_passes=0,
        )

    monkeypatch.setattr(module, "simulate_full_game", fake_simulate)
    reward_fn = module.build_fullgame_reward_fn(
        model=object(),
        tokenizer=object(),
        katago=_NoopKataGo(),
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        full_cfg=FullGameConfig(max_plies=6),
        seed=456,
    )

    reward_fn(prompts=[_prompt()], completions=["D4"])
    assert observed["max_plies"] == 6


def test_fullgame_reward_fn_aligns_board_turn_to_model_color(monkeypatch):
    module = _load_script_module("run_grpo_fullgame.py", "run_grpo_fullgame_script_alignment")
    monkeypatch.setenv("LOCAL_RANK", "0")

    observed = {"next_player": None, "model_color": None}

    def fake_simulate(**kwargs):
        observed["next_player"] = kwargs["board"].next_player()
        observed["model_color"] = kwargs["model_color"]
        return FullGameEpisodeResult(
            total_reward=0.0,
            scorelead_reward=0.0,
            terminal_reward=0.0,
            legality_reward_total=0.0,
            length_reward_total=0.0,
            model_turns=1,
            plies_played=1,
            truncated=False,
            terminated_by_illegal=False,
            pass_terminated=False,
            model_color=kwargs["model_color"],
            scorelead_delta_sum=0.0,
            final_outcome=0.0,
            katago_fallback_passes=0,
        )

    monkeypatch.setattr(module, "simulate_full_game", fake_simulate)
    reward_fn = module.build_fullgame_reward_fn(
        model=object(),
        tokenizer=object(),
        katago=_NoopKataGo(),
        reward_cfg=RewardConfig(),
        rollout_cfg=RolloutConfig(),
        full_cfg=FullGameConfig(model_color_mode="black"),
        seed=789,
    )

    prompt = "\n".join(
        [
            "board_size: 19",
            "to_play: B",
            "board:",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...................",
            "...X...............",
            "...................",
            "...................",
            "...................",
            "move:",
        ]
    )
    reward_fn(prompts=[prompt], completions=["Q16"])
    assert observed["model_color"] == "B"
    assert observed["next_player"] == "B"


def test_existing_run_grpo_still_exports_reward_builder():
    module = _load_script_module("run_grpo.py", "run_grpo_script_existing")
    assert hasattr(module, "build_reward_fn")
    assert callable(module.build_reward_fn)


def test_fullgame_reward_keeps_generated_position_and_side(monkeypatch):
    from types import SimpleNamespace
    from go_rl.board import BoardState
    module = _load_script_module("run_grpo_fullgame.py", "reward_unchanged_board")
    board = BoardState.empty(9)
    board.apply_move("B", "D4")
    observed = []

    def simulate(**kwargs):
        observed.append((kwargs["board"].to_prompt(), kwargs["model_color"]))
        return SimpleNamespace(total_reward=0, truncated=False, terminated_by_illegal=False, katago_fallback_passes=0)

    monkeypatch.setattr(module, "simulate_full_game", simulate)
    reward = module.build_fullgame_reward_fn(object(), object(), _NoopKataGo(), RewardConfig(),
        RolloutConfig(), FullGameConfig(seed_empty_board_prob=1, model_color_mode="black"), seed=123)
    reward([board.to_prompt()] * 2, ["F6", "F6"])
    assert observed == [(board.to_prompt(), "W")] * 2
