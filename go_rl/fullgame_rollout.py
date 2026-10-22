from __future__ import annotations

import logging
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from .board import BoardState
from .config import FullGameConfig, ReasoningConfig, RewardConfig, RolloutConfig
from .katago import KataGoResult
from .prompting import build_generation_inputs
from .model_interface import render_position
from .reasoning import (
    build_chat_template_kwargs,
    detect_reasoning_profile,
    strip_reasoning_for_reward,
)
from .rewards import is_legal_move, legality_reward
from .rollout import parse_move_answer
from .shared_types import KataGoService, board_key

logger = logging.getLogger(__name__)

@dataclass
class FullGameEpisodeResult:
    total_reward: float
    scorelead_reward: float
    terminal_reward: float
    legality_reward_total: float
    model_turns: int
    plies_played: int
    truncated: bool
    terminated_by_illegal: bool
    pass_terminated: bool
    model_color: str
    scorelead_delta_sum: float
    # Only a verified result (currently an illegal-action forfeit) is terminal.
    # A KataGo score estimate at a two-pass or capped endpoint is not a win.
    final_outcome: Optional[float]
    katago_fallback_passes: int
    estimated_outcome: Optional[float] = None
    outcome_source: str = "unverified"


def scorelead_from_model_perspective(score_lead: Optional[float], model_color: str) -> float:
    """Convert KataGo score lead (positive for black) into model perspective."""
    if score_lead is None:
        return 0.0
    if model_color.upper() == "B":
        return float(score_lead)
    return -float(score_lead)


def choose_model_color(board: BoardState, fullgame_cfg: FullGameConfig, episode_index: int) -> str:
    mode = fullgame_cfg.model_color_mode
    if mode == "black":
        return "B"
    elif mode == "white":
        return "W"
    return "B" if episode_index % 2 == 0 else "W"


def align_board_to_player_turn(board: BoardState, player: str) -> BoardState:
    """Return a copy of board where `player` is to move.

    If needed, inserts a single PASS by the current side to move.
    """
    aligned = board.clone()
    desired = player.strip().upper()
    if desired not in {"B", "W"}:
        raise ValueError(f"Unknown player color: {player}")
    to_move = aligned.next_player()
    if desired == to_move:
        return aligned
    aligned.apply_move(to_move, "PASS", strict=True)
    return aligned

def _extract_reward_move(completion: str, profile: str) -> str:
    """Return the answered move, or "" (illegal) when the answer is malformed."""
    return parse_move_answer(strip_reasoning_for_reward(completion, profile=profile))


def generate_model_move(
    model: object,
    tokenizer: object,
    board: BoardState,
    rollout_cfg: RolloutConfig,
    reasoning_profile: str,
    chat_template_kwargs: dict[str, object] | None = None,
    rules: str | dict[str, object] | None = None,
    komi: float | None = None,
    input_encoding: str = "compact",
) -> str:
    """Generate one model move from the current board prompt.

    Thinking controls belong in chat_template_kwargs; model.generate rejects them.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required at runtime
        logger.debug("Torch is unavailable during move generation; defaulting to PASS")
        return "PASS"

    prompt = render_position(board, input_encoding, rules=rules if rules is not None else "chinese",
                             komi=komi if komi is not None else 7.5)
    inputs = build_generation_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        chat_template_kwargs=chat_template_kwargs,
    )
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": rollout_cfg.max_tokens,
        "do_sample": True,
        "temperature": rollout_cfg.temperature,
        "top_p": rollout_cfg.top_p,
        "top_k": rollout_cfg.top_k,
        "pad_token_id": getattr(tokenizer, "eos_token_id", None),
        "use_cache": True,
    }

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
    finally:
        model.train(was_training)

    prompt_len = int(inputs["input_ids"].shape[-1])
    completion_tokens = output_ids[0][prompt_len:]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    return _extract_reward_move(completion_text, reasoning_profile)


def _apply_if_legal(board: BoardState, player: str, move: str, reward_cfg: RewardConfig) -> bool:
    """Apply a legal move and report whether it was played.

    The game continues against KataGo under real rules, so suicide and ko are illegal
    even at the basic legality level; applying them would corrupt the board.
    """
    if not is_legal_move(board, player, move, reward_cfg):
        return False
    try:
        board.apply_move(player, move, strict=True)
    except ValueError:
        return False
    return True


def simulate_full_game(
    board: BoardState,
    initial_completion: str,
    model: object,
    tokenizer: object,
    katago: KataGoService,
    reward_cfg: RewardConfig,
    rollout_cfg: RolloutConfig,
    fullgame_cfg: FullGameConfig,
    model_color: str,
    reasoning_cfg: ReasoningConfig | None = None,
    reasoning_model_name: str = "",
    analysis_cache: Optional[OrderedDict[tuple, KataGoResult]] = None,
    max_cache_size: Optional[int] = None,
    rules: str | dict[str, object] | None = None,
    komi: float | None = None,
    input_encoding: str = "compact",
) -> FullGameEpisodeResult:
    """Estimate the value of the first move with a sampled continuation.

    Only ``initial_completion`` is optimized by the caller's GRPO loss. Later
    model turns estimate that action's value; they are not training examples.
    This is root-action rollout optimization, not full-trajectory policy RL.
    """
    model_color = model_color.upper()
    if fullgame_cfg.termination_two_passes and board.has_two_pass_history():
        raise ValueError("Fullgame rollout root must precede the two-pass endpoint")
    if board.next_player() != model_color:
        raise ValueError(
            f"Board to-move ({board.next_player()}) does not match model_color ({model_color})."
        )
    opponent_color = "W" if model_color == "B" else "B"
    if reasoning_cfg is None:
        reasoning_cfg = ReasoningConfig(enabled=False)
    reasoning_profile = detect_reasoning_profile(
        model_name=reasoning_model_name,
        tokenizer=tokenizer,
        configured_profile=getattr(reasoning_cfg, "profile", "plain"),
    )
    chat_template_kwargs = build_chat_template_kwargs(reasoning_cfg, reasoning_profile)
    initial_board = board.clone()

    scorelead_delta_sum = 0.0
    legality_total = 0.0
    model_turns = 0
    plies_played = 0
    consecutive_passes = board.consecutive_passes()
    truncated = False
    terminated_by_illegal = False
    pass_terminated = False
    katago_fallback_passes = 0
    analysis_options = {}
    if rules is not None:
        analysis_options["rules"] = rules
    if komi is not None:
        analysis_options["komi"] = komi
    options_key = json.dumps(analysis_options, sort_keys=True)

    def _analyze_position(current_board: BoardState) -> KataGoResult:
        if analysis_cache is None:
            return katago.analyze_position(current_board, **analysis_options)
        key = (board_key(current_board), options_key)
        cached = analysis_cache.get(key)
        if cached is not None:
            analysis_cache.move_to_end(key)
            return cached
        result = katago.analyze_position(current_board, **analysis_options)
        analysis_cache[key] = result
        if max_cache_size is not None and max_cache_size > 0:
            while len(analysis_cache) > max_cache_size:
                analysis_cache.popitem(last=False)
        return result

    def _record_pass(move_text: str) -> None:
        nonlocal consecutive_passes
        if move_text.upper() == "PASS":
            consecutive_passes += 1
        else:
            consecutive_passes = 0

    def _terminal_due_to_passes() -> bool:
        return fullgame_cfg.termination_two_passes and consecutive_passes >= 2

    def _play_model_turn(move_text: str) -> None:
        nonlocal legality_total
        nonlocal model_turns
        nonlocal plies_played
        nonlocal terminated_by_illegal

        applied = _apply_if_legal(board, model_color, move_text, reward_cfg)
        # Reward legal formatting once, at the action actually being optimized.
        # A bonus at every turn would pay up to 30 points for surviving a loss,
        # overwhelming the former +/-1 outcome term.
        if not applied or model_turns == 0:
            legality_total += legality_reward(applied, reward_cfg)
        model_turns += 1
        if not applied:
            # Otherwise the model retries from the same position on its next turn.
            terminated_by_illegal = reward_cfg.illegal_move_ends_game
            return

        _record_pass(move_text)
        plies_played += 1

    first_move = _extract_reward_move(initial_completion, reasoning_profile)
    _play_model_turn(first_move)

    while not terminated_by_illegal:
        if _terminal_due_to_passes():
            pass_terminated = True
            break
        if plies_played >= fullgame_cfg.max_plies:
            truncated = True
            break
        if model_turns >= fullgame_cfg.max_model_turns:
            truncated = True
            break

        # Opponent (KataGo) turn.
        if board.next_player() == opponent_color:
            opponent_move = katago.choose_move(board, player=opponent_color, **analysis_options)
            if not is_legal_move(board, opponent_color, opponent_move, reward_cfg):
                opponent_move = "PASS"
                katago_fallback_passes += 1
            board.apply_move(opponent_color, opponent_move, strict=True)
            _record_pass(opponent_move)
            plies_played += 1
            if _terminal_due_to_passes():
                pass_terminated = True
                break
            if plies_played >= fullgame_cfg.max_plies:
                truncated = True
                break

        # Model turn.
        if board.next_player() == model_color:
            generated_move = generate_model_move(
                model=model,
                tokenizer=tokenizer,
                board=board,
                rollout_cfg=rollout_cfg,
                reasoning_profile=reasoning_profile,
                chat_template_kwargs=chat_template_kwargs,
                input_encoding=input_encoding,
                **analysis_options,
            )
            _play_model_turn(generated_move)

    final_outcome: Optional[float] = None
    estimated_outcome: Optional[float] = None
    outcome_source = "katago_estimate"
    if terminated_by_illegal:
        final_outcome = -1.0
        outcome_source = "illegal_action_forfeit"
    elif reward_cfg.scorelead_scale != 0:
        initial_lead = _analyze_position(initial_board).score_lead
        final_lead = _analyze_position(board).score_lead
        if initial_lead is None or final_lead is None:
            raise RuntimeError("Endpoint score reward requires KataGo scoreLead values.")
        initial_value = scorelead_from_model_perspective(initial_lead, model_color)
        final_value = scorelead_from_model_perspective(final_lead, model_color)
        # Include every transition, including the opponent's replies. This is
        # Phi(endpoint)-Phi(root), not a sum of model-only local blunder losses.
        # At a rollout cap it remains a bootstrap estimate, not a verified win.
        scorelead_delta_sum = final_value - initial_value
        estimated_outcome = float((final_value > 0) - (final_value < 0))
    else:
        outcome_source = "unverified"

    scorelead_reward = reward_cfg.scorelead_scale * scorelead_delta_sum
    terminal_reward = reward_cfg.terminal_outcome_bonus * (final_outcome or 0.0)
    total_reward = scorelead_reward + terminal_reward + legality_total
    return FullGameEpisodeResult(
        total_reward=total_reward,
        scorelead_reward=scorelead_reward,
        terminal_reward=terminal_reward,
        legality_reward_total=legality_total,
        model_turns=model_turns,
        plies_played=plies_played,
        truncated=truncated,
        terminated_by_illegal=terminated_by_illegal,
        pass_terminated=pass_terminated,
        model_color=model_color,
        scorelead_delta_sum=scorelead_delta_sum,
        final_outcome=final_outcome,
        katago_fallback_passes=katago_fallback_passes,
        estimated_outcome=estimated_outcome,
        outcome_source=outcome_source,
    )
