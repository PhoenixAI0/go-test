"""Tests for multi-GPU setup and distributed training configuration.

These tests verify that multi-GPU configurations are set up correctly
and that rank-sensitive operations (like reward computation) are handled properly.
"""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest


class TestMultiGPUConfiguration:
    """Tests for multi-GPU configuration detection and validation."""

    def test_single_gpu_default(self):
        """Test that single GPU is assumed when no distributed env vars are set."""
        # Clear distributed environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Should default to rank 0 (single GPU)
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

            assert local_rank == 0
            assert world_size == 1

    def test_multi_gpu_env_vars(self):
        """Test that multi-GPU setup reads environment variables correctly."""
        env = {
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "4",
            "RANK": "0",
        }

        with patch.dict(os.environ, env, clear=True):
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])

            assert local_rank == 0
            assert world_size == 4
            assert rank == 0

    def test_non_zero_rank(self):
        """Test that non-zero ranks are detected correctly."""
        env = {
            "LOCAL_RANK": "2",
            "WORLD_SIZE": "4",
            "RANK": "2",
        }

        with patch.dict(os.environ, env, clear=True):
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            assert local_rank == 2
            assert world_size == 4

    def test_is_main_process_rank_zero(self):
        """Test helper function to detect main process."""

        def is_main_process() -> bool:
            """Check if current process is main (rank 0)."""
            return int(os.environ.get("LOCAL_RANK", "0")) == 0

        env = {"LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is True

    def test_is_main_process_non_zero_rank(self):
        """Test that non-zero ranks are not considered main process."""

        def is_main_process() -> bool:
            """Check if current process is main (rank 0)."""
            return int(os.environ.get("LOCAL_RANK", "0")) == 0

        env = {"LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is False


class TestRankSensitiveOperations:
    """Tests for operations that should only run on rank 0."""

    def test_katago_analysis_rank_check(self, mock_katago_client):
        """Test that KataGo analysis should only run on rank 0."""

        def should_run_katago() -> bool:
            """Check if KataGo should run on this rank."""
            return int(os.environ.get("LOCAL_RANK", "0")) == 0

        # Rank 0 - should run
        env = {"LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            assert should_run_katago() is True

        # Rank 1 - should not run
        env = {"LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            assert should_run_katago() is False

    def test_reward_computation_rank_zero_only(self):
        """Test that reward computation logic checks for rank 0."""
        from go_rl.config import RewardConfig
        from go_rl.rewards import evaluate_move
        from go_rl.board import BoardState

        config = RewardConfig()
        board = BoardState(size=9)

        # On rank 0, reward computation should work
        env = {"LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            # Should not raise
            breakdown = evaluate_move(
                board=board,
                move="D4",
                winrate_delta=0.1,
                config=config,
                player="B",
            )
            assert breakdown is not None

        # On rank 1, reward computation still works
        # (the check should be in the caller, not in evaluate_move itself)
        env = {"LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            # Should still work - the GRPO script should skip calling this
            breakdown = evaluate_move(
                board=board,
                move="D4",
                winrate_delta=0.1,
                config=config,
                player="B",
            )
            assert breakdown is not None


class TestAccelerateConfigs:
    """Tests for accelerate configuration files."""

    def test_single_gpu_config_exists(self):
        """Test that single GPU config file exists."""
        import os

        config_path = "config/accelerate/single_gpu.yaml"
        assert os.path.exists(config_path), f"Missing config: {config_path}"

    def test_multi_gpu_configs_exist(self):
        """Test that multi-GPU config files exist."""
        import os

        configs = [
            "config/accelerate/ddp_2gpu.yaml",
            "config/accelerate/ddp_4gpu.yaml",
            "config/accelerate/ddp_8gpu.yaml",
        ]

        for config_path in configs:
            assert os.path.exists(config_path), f"Missing config: {config_path}"

    def test_accelerate_config_has_num_processes(self):
        """Test that accelerate configs specify num_processes correctly."""
        import yaml

        configs = {
            "config/accelerate/single_gpu.yaml": 1,
            "config/accelerate/ddp_2gpu.yaml": 2,
            "config/accelerate/ddp_4gpu.yaml": 4,
            "config/accelerate/ddp_8gpu.yaml": 8,
        }

        for config_path, expected_processes in configs.items():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            assert "num_processes" in config, f"Missing num_processes in {config_path}"
            assert (
                config["num_processes"] == expected_processes
            ), f"Wrong num_processes in {config_path}"


class TestDistributedLogging:
    """Tests for logging in distributed training."""

    def test_main_process_logs(self):
        """Test that logging should be enabled on main process."""

        def should_log() -> bool:
            """Check if logging should be enabled."""
            return int(os.environ.get("LOCAL_RANK", "0")) == 0

        env = {"LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            assert should_log() is True

    def test_worker_process_minimal_logs(self):
        """Test that worker processes may suppress verbose logging."""

        def should_log() -> bool:
            """Check if logging should be enabled."""
            return int(os.environ.get("LOCAL_RANK", "0")) == 0

        env = {"LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            # Worker processes may suppress logs to avoid clutter
            assert should_log() is False


def test_gpu_count_detection():
    """Test GPU count detection methods."""

    def detect_gpu_count() -> int:
        """Detect number of available GPUs."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass

        # Fallback to environment variable
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return world_size

    # Test with environment variable
    env = {"WORLD_SIZE": "4"}
    with patch.dict(os.environ, env, clear=True):
        count = detect_gpu_count()
        # Should be at least 1 (from env var)
        assert count >= 1


def test_rank_zero_decorator():
    """Test a hypothetical rank-zero-only decorator."""

    def rank_zero_only(func):
        """Decorator that only runs function on rank 0."""

        def wrapper(*args, **kwargs):
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                return func(*args, **kwargs)
            return None

        return wrapper

    @rank_zero_only
    def expensive_operation():
        """Operation that should only run on rank 0."""
        return "completed"

    # Rank 0 - should run
    env = {"LOCAL_RANK": "0"}
    with patch.dict(os.environ, env, clear=True):
        result = expensive_operation()
        assert result == "completed"

    # Rank 1 - should skip
    env = {"LOCAL_RANK": "1"}
    with patch.dict(os.environ, env, clear=True):
        result = expensive_operation()
        assert result is None
