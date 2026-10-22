import math

import pytest

torch = pytest.importorskip("torch")

from go_rl.policy_optimization import (
    categorical_ppo_loss,
    categorical_statistics,
    generalized_advantages,
    masked_log_probs,
    regularized_policy_target,
    soft_policy_cross_entropy,
)


def test_masked_policy_normalizes_sampling_likelihood_and_entropy():
    logits = torch.tensor([[1.0, 1000.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    log_probs = masked_log_probs(logits, mask, temperature=0.5)
    expected = torch.softmax(torch.tensor([2.0, 4.0]), 0)
    assert torch.allclose(log_probs.exp()[0, [0, 2]], expected)
    assert log_probs.exp()[0, 1] == 0
    chosen, entropy = categorical_statistics(logits, torch.tensor([2]), mask, 0.5)
    assert torch.allclose(chosen.exp(), expected[1:])
    assert entropy.item() == pytest.approx(-(expected * expected.log()).sum().item())
    (-chosen.mean()).backward()
    assert logits.grad[0, 1] == 0


def test_illegal_stored_action_or_no_legal_action_fails():
    with pytest.raises(ValueError, match="illegal"):
        categorical_statistics(torch.zeros(1, 2), torch.tensor([1]), torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="at least one"):
        masked_log_probs(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_regularized_target_matches_closed_form_and_detaches():
    logits = torch.tensor([[math.log(0.2), math.log(0.8), 2.0]], requires_grad=True)
    q = torch.tensor([[0.2, -0.4, 100.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    target = regularized_policy_target(logits, q, mask, kl_coefficient=0.3, entropy_coefficient=0.2)
    expected = torch.softmax((q[0, :2] + 0.3 * logits[0, :2]) / 0.5, 0)
    assert torch.allclose(target[0, :2], expected)
    assert target[0, 2] == 0
    assert not target.requires_grad
    # A constant baseline cannot change the improvement policy.
    shifted = regularized_policy_target(logits, q + 10, mask, kl_coefficient=0.3, entropy_coefficient=0.2)
    assert torch.allclose(target, shifted, atol=2e-6)


def test_regularized_target_limits_and_soft_target_gradients():
    logits = torch.tensor([[1.0, -1.0, 10.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    no_change = regularized_policy_target(logits, torch.zeros_like(logits), mask,
                                          kl_coefficient=1.0, entropy_coefficient=0.0)
    assert torch.allclose(no_change, masked_log_probs(logits, mask).exp())
    uniform = regularized_policy_target(logits, torch.zeros_like(logits), mask,
                                       kl_coefficient=0.0, entropy_coefficient=0.5)
    assert torch.allclose(uniform, torch.tensor([[0.5, 0.5, 0.0]]))
    loss = soft_policy_cross_entropy(logits, uniform, mask)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0


def test_alternating_player_monte_carlo_returns_flip_signs():
    result = generalized_advantages(
        rewards=torch.tensor([0.0, 0.0, 1.0]), values=torch.zeros(3),
        next_values=torch.zeros(3), terminated=torch.tensor([False, False, True]),
        truncated=torch.zeros(3, dtype=torch.bool), next_player_sign=-torch.ones(3),
        gamma=1, gae_lambda=1,
    )
    assert torch.equal(result.returns, torch.tensor([1.0, -1.0, 1.0]))
    assert torch.equal(result.advantages, result.returns)


def test_truncation_bootstraps_but_does_not_leak_next_episode():
    result = generalized_advantages(
        rewards=torch.tensor([0.0, 5.0]), values=torch.tensor([0.2, 0.1]),
        next_values=torch.tensor([0.7, 100.0]), terminated=torch.tensor([False, True]),
        truncated=torch.tensor([True, False]), next_player_sign=torch.tensor([-1.0, -1.0]),
        gamma=1, gae_lambda=1,
    )
    assert torch.allclose(result.returns, torch.tensor([-0.7, 5.0]))
    assert torch.allclose(result.advantages, torch.tensor([-0.9, 4.9]))


def test_batched_gae_lambda_and_same_player_transitions():
    rewards = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    result = generalized_advantages(rewards, torch.zeros_like(rewards), torch.zeros_like(rewards),
        terminated=torch.tensor([[False, False], [True, True]]),
        truncated=torch.zeros_like(rewards, dtype=torch.bool), gamma=1, gae_lambda=0.5)
    assert torch.allclose(result.returns, torch.tensor([[0.5, -0.5], [1.0, -1.0]]))


@pytest.mark.parametrize("advantage,gradient_zero", [(1.0, True), (-1.0, False)])
def test_ppo_clipping_retains_correct_sign_dependent_gradient(advantage, gradient_zero):
    logits = torch.tensor([[math.log(0.8), math.log(0.2)]], requires_grad=True)
    result = categorical_ppo_loss(
        logits, torch.zeros(1), torch.tensor([0]), torch.ones_like(logits, dtype=torch.bool),
        old_log_probs=torch.tensor([math.log(0.4)]), advantages=torch.tensor([advantage]),
        returns=torch.zeros(1), value_coefficient=0, entropy_coefficient=0,
    )
    result.total.backward()
    assert result.policy.item() == pytest.approx(-1.2 if advantage > 0 else 2.0)
    assert (logits.grad.abs().sum() == 0).item() is gradient_zero
    assert result.clip_fraction.item() == 1.0
    assert result.approximate_kl.item() >= 0


def test_identity_ppo_ratio_uses_temperature_mask_and_detached_targets():
    logits = torch.tensor([[2.0, 1.0, 100.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    actions = torch.tensor([1])
    old, _ = categorical_statistics(logits.detach(), actions, mask, temperature=0.7)
    values = torch.tensor([0.3], requires_grad=True)
    returns = torch.tensor([0.8], requires_grad=True)
    advantages = torch.tensor([0.5], requires_grad=True)
    result = categorical_ppo_loss(logits, values, actions, mask, old, advantages, returns,
                                  temperature=0.7, entropy_coefficient=0)
    result.total.backward()
    assert result.approximate_kl.item() == pytest.approx(0)
    assert result.clip_fraction.item() == 0
    assert returns.grad is None and advantages.grad is None
    assert values.grad.item() < 0
    assert logits.grad[0, 1].item() < 0
    assert logits.grad[0, 2].item() == 0
