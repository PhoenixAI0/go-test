"""Tests for GRPO reward computation.

These tests verify that reward functions work correctly and integrate
properly with KataGo analysis.
"""

from __future__ import annotations

from go_rl.board import BoardState
from go_rl.config import RewardConfig
from go_rl.rewards import (
    RewardBreakdown,
    evaluate_move,
    legality_reward,
    length_reward,
    outcome_reward,
    winrate_reward,
)


class TestRewardComponents:
    """Tests for individual reward components."""

    def test_winrate_reward_positive_delta(self):
        """Test positive winrate delta gives positive reward."""
        config = RewardConfig(winrate_scale=2.0)
        delta = 0.1  # Winrate improved by 10%

        reward = winrate_reward(delta, config)
        assert reward == 0.2  # 0.1 * 2.0

    def test_winrate_reward_negative_delta(self):
        """Test negative winrate delta gives negative reward."""
        config = RewardConfig(winrate_scale=2.0)
        delta = -0.1  # Winrate declined by 10%

        reward = winrate_reward(delta, config)
        assert reward == -0.2

    def test_winrate_reward_zero_scale(self):
        """Test zero scale disables winrate component."""
        config = RewardConfig(winrate_scale=0.0)
        delta = 0.5

        reward = winrate_reward(delta, config)
        assert reward == 0.0

    def test_legality_reward_legal_move(self):
        """Test legal moves get bonus."""
        config = RewardConfig(legal_bonus=0.2)

        reward = legality_reward(is_legal=True, config=config)
        assert reward == 0.2

    def test_legality_reward_illegal_move(self):
        """Test illegal moves get penalty."""
        config = RewardConfig(illegal_penalty=-1.0)

        reward = legality_reward(is_legal=False, config=config)
        assert reward == -1.0

    def test_length_reward_pass(self):
        """Test PASS moves have no length penalty."""
        config = RewardConfig(length_penalty=-0.05)

        reward = length_reward("PASS", config)
        assert reward == 0.0

    def test_length_reward_short_move(self):
        """Test short moves have no length penalty."""
        config = RewardConfig(length_penalty=-0.05)

        # Moves like "D4" (3 chars or less) should not be penalized
        reward = length_reward("D4", config)
        assert reward == 0.0

    def test_length_reward_long_output(self):
        """Test long output strings are penalized."""
        config = RewardConfig(length_penalty=-0.05)

        # Long strings (e.g., model rambling) should be penalized
        reward = length_reward("This is a very long output", config)
        assert reward == -0.05

    def test_outcome_reward_none(self):
        """Test no outcome gives zero reward."""
        config = RewardConfig(full_game_bonus=0.5)

        reward = outcome_reward(result=None, config=config)
        assert reward == 0.0

    def test_outcome_reward_win(self):
        """Test winning gives positive reward."""
        config = RewardConfig(full_game_bonus=0.5)

        reward = outcome_reward(result=1.0, config=config)
        assert reward == 0.5

    def test_outcome_reward_loss(self):
        """Test losing gives negative reward."""
        config = RewardConfig(full_game_bonus=0.5)

        reward = outcome_reward(result=-1.0, config=config)
        assert reward == -0.5


class TestEvaluateMove:
    """Tests for the complete evaluate_move function."""

    def test_legal_move_evaluation(self):
        """Test evaluating a legal move."""
        config = RewardConfig(
            winrate_scale=1.0,
            legal_bonus=0.2,
            length_penalty=-0.05,
        )

        board = BoardState(size=9)
        move = "D4"  # Legal opening move
        winrate_delta = 0.05

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=winrate_delta,
            config=config,
            player="B",
        )

        # Check individual components
        assert breakdown.winrate == 0.05  # 0.05 * 1.0
        assert breakdown.legality == 0.2  # Legal bonus
        assert breakdown.length == 0.0  # Short move, no penalty
        assert breakdown.outcome == 0.0  # No game result

        # Check total
        expected_total = 0.05 + 0.2
        assert breakdown.total == expected_total

    def test_illegal_move_evaluation(self):
        """Test evaluating an illegal move."""
        config = RewardConfig(
            winrate_scale=1.0,
            legal_bonus=0.2,
            illegal_penalty=-1.0,
        )

        board = BoardState(size=9)
        board.apply_move("B", "D4")

        # Try to play on occupied spot
        move = "D4"
        winrate_delta = 0.0

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=winrate_delta,
            config=config,
            player="W",
        )

        # Should get illegal penalty
        assert breakdown.legality == -1.0
        assert breakdown.total < 0.0

    def test_strict_legality_checking(self):
        """Test strict legality mode."""
        config = RewardConfig(
            legal_bonus=0.2,
            illegal_penalty=-1.0,
            strict_legality=True,
        )

        board = BoardState(size=9)
        move = "D4"

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=0.0,
            config=config,
            player="B",
        )

        # Should use strict legality checking
        assert breakdown.legality == 0.2  # Legal move

    def test_basic_legality_checking(self):
        """Test basic legality mode (default)."""
        config = RewardConfig(
            legal_bonus=0.2,
            illegal_penalty=-1.0,
            strict_legality=False,
        )

        board = BoardState(size=9)
        move = "D4"

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=0.0,
            config=config,
            player="B",
        )

        # Should use basic legality checking
        assert breakdown.legality == 0.2

    def test_game_outcome_win(self):
        """Test evaluation with winning game outcome."""
        config = RewardConfig(
            winrate_scale=0.0,
            legal_bonus=0.0,
            full_game_bonus=1.0,
        )

        board = BoardState(size=9)
        move = "PASS"

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=0.0,
            config=config,
            game_result=1.0,  # Win
            player="B",
        )

        # Should get full game bonus for winning
        assert breakdown.outcome == 1.0
        assert breakdown.total == 1.0

    def test_combined_rewards(self):
        """Test all reward components combine correctly."""
        config = RewardConfig(
            winrate_scale=2.0,
            legal_bonus=0.2,
            illegal_penalty=-1.0,
            length_penalty=-0.05,
            full_game_bonus=0.5,
        )

        board = BoardState(size=9)
        move = "D4"
        winrate_delta = 0.1  # Improved by 10%

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=winrate_delta,
            config=config,
            player="B",
        )

        # Expected components:
        # - Winrate: 0.1 * 2.0 = 0.2
        # - Legality: 0.2 (legal)
        # - Length: 0.0 (short move)
        # - Outcome: 0.0 (no result)
        # Total: 0.4

        assert breakdown.winrate == 0.2
        assert breakdown.legality == 0.2
        assert breakdown.length == 0.0
        assert breakdown.outcome == 0.0
        assert breakdown.total == 0.4

    def test_invalid_move_format(self):
        """Test handling of invalid move format."""
        config = RewardConfig(illegal_penalty=-1.0)

        board = BoardState(size=9)
        move = "INVALID_MOVE_123"

        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=0.0,
            config=config,
            player="B",
        )

        # Should treat as illegal
        assert breakdown.legality == -1.0


class TestRewardBreakdown:
    """Tests for RewardBreakdown dataclass."""

    def test_reward_breakdown_total(self):
        """Test that total property sums components correctly."""
        breakdown = RewardBreakdown(
            winrate=0.1,
            legality=0.2,
            length=-0.05,
            outcome=0.5,
        )

        expected_total = 0.1 + 0.2 - 0.05 + 0.5
        assert breakdown.total == expected_total

    def test_reward_breakdown_all_zeros(self):
        """Test breakdown with all zero components."""
        breakdown = RewardBreakdown(
            winrate=0.0,
            legality=0.0,
            length=0.0,
            outcome=0.0,
        )

        assert breakdown.total == 0.0

    def test_reward_breakdown_negative_total(self):
        """Test breakdown with negative total."""
        breakdown = RewardBreakdown(
            winrate=-0.3,
            legality=-1.0,
            length=-0.05,
            outcome=0.0,
        )

        assert breakdown.total < 0.0


class TestKataGoIntegration:
    """Tests for integration with KataGo client."""

    def test_evaluate_with_katago_winrate(self, mock_katago_client):
        """Test evaluating move with KataGo-provided winrate."""
        config = RewardConfig(winrate_scale=1.0)
        board = BoardState(size=9)

        # Get initial winrate from KataGo
        initial_result = mock_katago_client.analyze_position(board)
        initial_winrate = initial_result.winrate

        # Apply move and get new winrate
        move = "D4"
        board.apply_move("B", move)
        new_result = mock_katago_client.analyze_position(board)
        new_winrate = new_result.winrate

        # Compute winrate delta
        winrate_delta = new_winrate - initial_winrate

        # Evaluate move
        breakdown = evaluate_move(
            board=board,
            move=move,
            winrate_delta=winrate_delta,
            config=config,
            player="B",
        )

        # Winrate component should match delta
        assert abs(breakdown.winrate - winrate_delta) < 0.01


def test_reward_config_defaults():
    """Test RewardConfig default values."""
    config = RewardConfig()

    assert config.winrate_scale == 1.0
    assert config.legal_bonus == 0.2
    assert config.illegal_penalty == -1.0
    assert config.length_penalty == -0.05
    assert config.full_game_bonus == 0.5
    assert config.strict_legality is False
