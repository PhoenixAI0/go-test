"""On-policy categorical self-play collection and PPO updates.

Every move is sampled from the model, including the opponent's moves. Values
and terminal rewards use the acting player's perspective. The GTP referee
checks actions and adjudicates two-pass games; a capped game has no outcome.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

from .board import BoardState
from .data_prep import state_record_from_board
from .game_result import parse_game_result
from .model_interface import action_to_move, legal_action_mask
from .policy_model import collate_policy_inputs, encode_policy_position
from .policy_optimization import categorical_ppo_loss, generalized_advantages, masked_log_probs
from .rules import legality_superko, normalize_rules
from .seed_positions import replay_seed_board
from .scoring import terminal_board_area_result, validate_board_area_rules


class Referee(Protocol):
    def send(self, command: str, timeout: float = 600.0) -> str: ...


@dataclass
class PolicyTransition:
    input_ids: list[int]
    legal_mask: list[bool]
    action: int
    player: str
    old_log_prob: float
    value: float
    reward: float = 0.0
    next_value: float = 0.0
    terminated: bool = False
    truncated: bool = False


@dataclass
class PolicyEpisode:
    transitions: list[PolicyTransition] = field(default_factory=list)
    moves: list[tuple[str, str]] = field(default_factory=list)
    black_outcome: float | None = None
    final_score: str | None = None
    outcome_source: str = "value_bootstrap_at_cap"
    temperature: float = 1.0
    start_position: dict[str, Any] | None = None
    raw_board_result: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "plies": len(self.transitions), "moves": self.moves,
            "terminated": self.transitions[-1].terminated if self.transitions else False,
            "truncated": self.transitions[-1].truncated if self.transitions else False,
            "black_outcome": self.black_outcome, "final_score": self.final_score,
            "outcome_source": self.outcome_source,
            "bootstrap_value": self.transitions[-1].next_value if self.transitions else None,
            "behavior_temperature": self.temperature,
            "start_position": self.start_position,
            "starting_plies": len(self.start_position["moves"]) if self.start_position else 0,
            "raw_board_result": self.raw_board_result,
        }


def referee_black_outcome(score: str) -> float:
    """Interpret an adjudicated GTP result, rejecting unscored/void games."""
    return parse_game_result(score)[1]


@torch.no_grad()
def collect_selfplay_episode(
    model: Any, tokenizer: Any, referee: Referee, *, max_plies: int = 64,
    encoding: str = "compact", rules: Any = "koPOSITIONALscoreAREAtaxNONEsui0",
    komi: float = 7.5, temperature: float = 1.0, start_board: BoardState | None = None,
) -> PolicyEpisode:
    """Collect model turns from empty or an explicitly provided training seed.

    A rules mismatch or referee rejection aborts collection instead of becoming
    a fictitious model loss. ``final_score`` is an engine-adjudicated result,
    not a proof that every dead group has been independently solved.
    """
    if max_plies < 1 or not math.isfinite(komi):
        raise ValueError("max_plies must be positive and komi finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # This collector has one two-pass phase. Territory encore phases, button
    # turns and friendly-pass shortcuts need a different terminal-state model;
    # accepting their GTP final_score here could promote an unfinished estimate
    # into a terminal reward. Restrict collection to the audited area objective.
    validate_board_area_rules(rules)
    superko = legality_superko(rules)
    canonical_rules = json.dumps(normalize_rules(rules), separators=(",", ":"))
    board = (replay_seed_board(start_board, board_size=model.board_size, rules=rules)
             if start_board is not None else BoardState(size=model.board_size))
    referee.send(f"boardsize {board.size}")
    referee.send("clear_board")
    referee.send(f"komi {komi:g}")
    # Required, not try_send: unsupported rules must never silently fall back.
    referee.send(f"kata-set-rules {canonical_rules}")
    if start_board is not None:
        from .board import coords_to_gtp
        for player, coords in board.moves:
            referee.send(f"play {player.lower()} {coords_to_gtp(coords, board.size)}")
    model.eval()  # Dropout must not alter the behavior policy or PPO ratios.
    episode = PolicyEpisode(temperature=temperature,
                            start_position=state_record_from_board(board) if start_board is not None else None)

    def forward_position():
        ids = encode_policy_position(board, tokenizer, encoding, rules=rules, komi=komi)
        legal = legal_action_mask(board, superko=superko)
        mask = torch.tensor([legal], dtype=torch.bool, device=model.device)
        output = model(**collate_policy_inputs([ids], tokenizer, model.device), legal_mask=mask)
        if not bool(torch.isfinite(output.value).all()):
            raise ValueError("Nonfinite model value during collection")
        return ids, legal, mask, output

    for _ in range(max_plies):
        ids, legal, mask, output = forward_position()
        log_probs = masked_log_probs(output.logits, mask, temperature)
        action = int(torch.multinomial(log_probs[0].exp(), 1).item())
        value = float(output.value[0].item())
        if episode.transitions:
            episode.transitions[-1].next_value = value
        player = board.next_player()
        move = action_to_move(action, board.size)
        # Both the local mask and the referee must agree this action is legal.
        referee.send(f"play {player.lower()} {move}")
        board.apply_move(player, move, strict=True)
        transition = PolicyTransition(ids, legal, action, player,
                                      float(log_probs[0, action].item()), value)
        episode.transitions.append(transition)
        episode.moves.append((player, move))
        if board.consecutive_passes() >= 2:
            episode.final_score = referee.send("final_score").strip()
            episode.black_outcome = referee_black_outcome(episode.final_score)
            episode.outcome_source = "katago_gtp_final_score"
            try:
                validate_board_area_rules(rules)
            except ValueError:
                pass  # The raw-area diagnostic does not implement other scores.
            else:
                # Diagnostic only: KataGo's deterministic pass-alive scoring can
                # differ from counting every remaining stone as its own color.
                episode.raw_board_result = terminal_board_area_result(board, komi=komi, rules=rules)
            transition.reward = episode.black_outcome * (1 if player == "B" else -1)
            transition.terminated = True
            break
    else:
        # Value is in the NEXT player's perspective; signed GAE converts it.
        _, _, _, output = forward_position()
        episode.transitions[-1].next_value = float(output.value[0].item())
        episode.transitions[-1].truncated = True
    return episode


def optimize_selfplay_episodes(
    model: Any, tokenizer: Any, optimizer: torch.optim.Optimizer,
    episodes: Sequence[PolicyEpisode], *, minibatch_size: int = 8,
    epochs: int = 1, gamma: float = 1.0, gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2, value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01, max_grad_norm: float = 1.0,
    normalize_advantages: bool = False, target_kl: float | None = 0.03,
) -> dict[str, float | int]:
    """Optimize all collected moves; fixed behavior policy until next collection.

    Retaining evaluation mode still allows gradients and makes likelihood ratios
    reproducible when a backbone has dropout. PPO epochs reuse the stored policy
    probabilities, never recompute them from the updated model.
    """
    if not episodes or any(not episode.transitions for episode in episodes):
        raise ValueError("Need nonempty episodes")
    if minibatch_size < 1 or epochs < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("minibatch_size, epochs and max_grad_norm must be positive")
    if target_kl is not None and (not math.isfinite(target_kl) or target_kl <= 0):
        raise ValueError("target_kl must be positive or None")
    temperatures = {episode.temperature for episode in episodes}
    if len(temperatures) != 1:
        raise ValueError("A PPO batch must use one recorded behavior temperature")
    temperature = temperatures.pop()
    transitions = [step for episode in episodes for step in episode.transitions]
    advantage_parts, return_parts = [], []
    for episode in episodes:
        steps = episode.transitions
        if not (steps[-1].terminated or steps[-1].truncated):
            raise ValueError("Collected episode needs an explicit terminal or truncation boundary")
        if any(a.player == b.player for a, b in zip(steps, steps[1:])):
            raise ValueError("Self-play trajectories must alternate player perspective")
        scalar = lambda name: torch.tensor([getattr(step, name) for step in steps], dtype=torch.float32)
        estimates = generalized_advantages(
            scalar("reward"), scalar("value"), scalar("next_value"),
            torch.tensor([step.terminated for step in steps]),
            torch.tensor([step.truncated for step in steps]),
            next_player_sign=-torch.ones(len(steps)), gamma=gamma, gae_lambda=gae_lambda,
        )
        advantage_parts.append(estimates.advantages)
        return_parts.append(estimates.returns)
    advantages, returns = torch.cat(advantage_parts), torch.cat(return_parts)
    if normalize_advantages and len(transitions) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    totals = dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0, approximate_kl=0.0,
                  clip_fraction=0.0, gradient_norm=0.0)
    samples_seen, batches = 0, 0
    stopped_for_kl = False
    for epoch in range(epochs):
        for indices in torch.randperm(len(transitions)).split(minibatch_size):
            batch = [transitions[index] for index in indices.tolist()]
            inputs = collate_policy_inputs([step.input_ids for step in batch], tokenizer, model.device)
            legal = torch.tensor([step.legal_mask for step in batch], dtype=torch.bool, device=model.device)
            output = model(**inputs, legal_mask=legal)
            losses = categorical_ppo_loss(
                output.logits, output.value,
                torch.tensor([step.action for step in batch], dtype=torch.long, device=model.device),
                legal, torch.tensor([step.old_log_prob for step in batch], device=model.device),
                advantages[indices].to(model.device), returns[indices].to(model.device),
                clip_epsilon=clip_epsilon, value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient, temperature=temperature,
            )
            # Every collected move receives one update. The KL budget limits
            # additional reuse of the same behavior data in subsequent epochs.
            if epoch > 0 and target_kl is not None and float(losses.approximate_kl.item()) > target_kl:
                stopped_for_kl = True
                break
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            count = len(batch)
            observed = dict(loss=losses.total, policy_loss=losses.policy, value_loss=losses.value,
                            entropy=losses.entropy, approximate_kl=losses.approximate_kl,
                            clip_fraction=losses.clip_fraction, gradient_norm=norm)
            for key, value in observed.items():
                totals[key] += float(value.detach().item()) * count
            samples_seen += count
            batches += 1
        if stopped_for_kl:
            break
    return {**{key: value / max(samples_seen, 1) for key, value in totals.items()},
            "collected_moves": len(transitions), "optimized_move_samples": samples_seen,
            "optimizer_steps": batches, "stopped_for_kl": int(stopped_for_kl),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std(unbiased=False)),
            "return_mean": float(returns.mean())}
