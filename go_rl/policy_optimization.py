"""Finite-action objectives and trajectory returns for a Go policy.

All values/rewards use the acting player's perspective. Collection must retain
the legal mask, actual behavior log probability, and true termination versus
rollout truncation. These primitives do not turn teacher estimates into wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def masked_log_probs(
    logits: torch.Tensor, legal_mask: torch.Tensor, temperature: float = 1.0,
) -> torch.Tensor:
    """The same normalized legal policy used for both sampling and training."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if logits.ndim < 1 or logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have equal [..., action] shape")
    if legal_mask.dtype != torch.bool:
        raise ValueError("legal_mask must be boolean")
    if not legal_mask.any(dim=-1).all():
        raise ValueError("Every position must have at least one legal action")
    if not torch.isfinite(logits[legal_mask]).all():
        raise ValueError("Legal-action logits must be finite")
    return torch.log_softmax((logits.float() / temperature).masked_fill(~legal_mask, -torch.inf), dim=-1)


def categorical_statistics(
    logits: torch.Tensor, actions: torch.Tensor, legal_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return chosen-action log probability and exact legal-action entropy."""
    if actions.shape != logits.shape[:-1] or actions.dtype != torch.long:
        raise ValueError("actions must be integer indices with the logits batch shape")
    if ((actions < 0) | (actions >= logits.shape[-1])).any():
        raise ValueError("Action index out of range")
    log_probs = masked_log_probs(logits, legal_mask, temperature)
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(selected).all():
        raise ValueError("Stored action is illegal under its stored mask")
    entropy = -(log_probs.exp() * log_probs.masked_fill(~legal_mask, 0)).sum(dim=-1)
    return selected, entropy


@torch.no_grad()
def regularized_policy_target(
    policy_logits: torch.Tensor, action_values: torch.Tensor,
    legal_mask: torch.Tensor, *, kl_coefficient: float = 0.1,
    entropy_coefficient: float = 0.03,
) -> torch.Tensor:
    """Solve max_p E_p[Q] - beta KL(p||prior) + alpha H(p) exactly.

    This is the KLENT finite-action improvement operator. A teacher Q makes it
    teacher-guided improvement, not reproduction of KLENT's learned-Q self-play.
    The returned target is detached for fitting the policy by cross-entropy.
    """
    coefficients = (kl_coefficient, entropy_coefficient)
    if any(not math.isfinite(c) or c < 0 for c in coefficients) or sum(coefficients) <= 0:
        raise ValueError("Regularization coefficients must be finite, nonnegative, and sum positive")
    if action_values.shape != policy_logits.shape:
        raise ValueError("action_values must match policy_logits")
    log_prior = masked_log_probs(policy_logits, legal_mask)
    if not torch.isfinite(action_values[legal_mask]).all():
        raise ValueError("Legal action values must be finite")
    # Replace masked -inf before multiplying by beta, which may equal zero.
    target_logits = (action_values.float().masked_fill(~legal_mask, 0)
                     + kl_coefficient * log_prior.masked_fill(~legal_mask, 0)) / sum(coefficients)
    return masked_log_probs(target_logits, legal_mask).exp()


def soft_policy_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy against a detached, validated legal target policy."""
    if targets.shape != logits.shape or not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Targets must be finite nonnegative distributions matching logits")
    if targets[~legal_mask].abs().sum() > 1e-7:
        raise ValueError("Target policy assigns mass to illegal actions")
    if not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError("Target policy must sum to one")
    log_probs = masked_log_probs(logits, legal_mask).masked_fill(~legal_mask, 0)
    return -(targets.detach() * log_probs).sum(-1).mean()


@dataclass
class TrajectoryReturns:
    advantages: torch.Tensor
    returns: torch.Tensor


@torch.no_grad()
def generalized_advantages(
    rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
    terminated: torch.Tensor, truncated: torch.Tensor, *,
    next_player_sign: torch.Tensor | None = None,
    gamma: float = 1.0, gae_lambda: float = 0.95,
) -> TrajectoryReturns:
    """GAE for [time, ...] data with explicit player and episode boundaries.

    ``next_player_sign[t]`` is -1 if V(s[t+1]) is in the opponent's
    perspective and +1 if it is in the same player's perspective. Truncation
    bootstraps from next_values, but stops recursion into the next episode.
    True termination suppresses both bootstrap and recursion. Callers collecting
    one full model/opponent turn per transition use +1 rather than alternating.
    """
    tensors = (values, next_values, terminated, truncated)
    if rewards.ndim < 1 or rewards.shape[0] == 0 or any(t.shape != rewards.shape for t in tensors):
        raise ValueError("Trajectory tensors must share a nonempty [time, ...] shape")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("terminated and truncated must be boolean")
    if (terminated & truncated).any():
        raise ValueError("A transition cannot be both terminal and truncated")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not all(torch.isfinite(t).all() for t in (rewards, values, next_values)):
        raise ValueError("Rewards and values must be finite")
    signs = torch.ones_like(values) if next_player_sign is None else next_player_sign
    if signs.shape != rewards.shape or not ((signs == 1) | (signs == -1)).all():
        raise ValueError("next_player_sign must match rewards and contain only -1 or +1")
    bootstrap = (~terminated).to(values.dtype)
    recurse = (~(terminated | truncated)).to(values.dtype)
    deltas = rewards + gamma * bootstrap * signs * next_values - values
    advantages = torch.zeros_like(deltas)
    tail = torch.zeros_like(deltas[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        tail = deltas[index] + gamma * gae_lambda * recurse[index] * signs[index] * tail
        advantages[index] = tail
    return TrajectoryReturns(advantages=advantages, returns=advantages + values)


@dataclass
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def categorical_ppo_loss(
    logits: torch.Tensor, values: torch.Tensor, actions: torch.Tensor,
    legal_mask: torch.Tensor, old_log_probs: torch.Tensor,
    advantages: torch.Tensor, returns: torch.Tensor, *,
    clip_epsilon: float = 0.2, value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01, temperature: float = 1.0,
) -> PPOLoss:
    """PPO over stored Go actions, with exact categorical entropy.

    old_log_probs must be actual collection probabilities after the same mask
    and temperature; no teacher-selected/search action may masquerade as on-policy.
    This loss is a clipped surrogate, not an unbiased off-policy return estimator.
    """
    if not math.isfinite(clip_epsilon) or not 0 <= clip_epsilon < 1:
        raise ValueError("clip_epsilon must be finite and in [0, 1)")
    if any(not math.isfinite(c) or c < 0 for c in (value_coefficient, entropy_coefficient)):
        raise ValueError("Loss coefficients must be finite and nonnegative")
    expected = logits.shape[:-1]
    if any(t.shape != expected for t in (values, old_log_probs, advantages, returns)):
        raise ValueError("Values, log probabilities, advantages and returns must match batch shape")
    if not all(torch.isfinite(t).all() for t in (values, old_log_probs, advantages, returns)):
        raise ValueError("PPO inputs must be finite")
    if (old_log_probs > 1e-6).any():
        raise ValueError("Behavior log probabilities must be nonpositive")
    log_probs, entropies = categorical_statistics(logits, actions, legal_mask, temperature)
    log_ratio = log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Importance ratio overflow; behavior data is too far off-policy")
    advantage = advantages.detach()
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
    value = 0.5 * (values - returns.detach()).square().mean()
    entropy = entropies.mean()
    total = policy + value_coefficient * value - entropy_coefficient * entropy
    with torch.no_grad():
        approximate_kl = (ratio - 1 - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > clip_epsilon).float().mean()
    return PPOLoss(total, policy, value, entropy, approximate_kl, clip_fraction)
