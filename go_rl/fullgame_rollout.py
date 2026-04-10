from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import MutableMapping, Optional, Protocol

from .board import BoardState
from .config import FullGameConfig, RewardConfig, RolloutConfig
from .katago import KataGoResult
from .prompting import build_generation_inputs, build_prompt, parse_move
from .reasoning import (
    build_chat_template_kwargs,
    build_generation_kwargs,
    detect_reasoning_profile,
    strip_reasoning_for_reward,
)
from .rewards import (
    is_legal_move,
    legality_reward,
    length_reward,
    resolve_legality_level,
)
from .rollout import normalize_move
from .rules import LegalityLevel

logger = logging.getLogger(__name__)


class _KataGoService(Protocol):
    def analyze_position(self, board: BoardState) -> KataGoResult: ...
    def choose_move(self, board: BoardState, player: str | None = None) -> str: ...


@dataclass
class FullGameEpisodeResult:
    total_reward: float
    scorelead_reward: float
    terminal_reward: float
    legality_reward_total: float
    length_reward_total: float
    model_turns: int
    plies_played: int
    truncated: bool
    terminated_by_illegal: bool
    pass_terminated: bool
    model_color: str
    scorelead_delta_sum: float
    final_outcome: float
    katago_fallback_passes: int


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


def board_key(
    board: BoardState,
) -> tuple[int, str, tuple[tuple[int, ...], ...]]:
    return (
        board.size,
        board.next_player(),
        tuple(tuple(row) for row in board.current_board()),
    )


def _parse_completion_to_move(completion: str, stop_at_pass: bool) -> str:
    stripped = completion.strip()
    if not stripped:
        return "PASS"
    try:
        return parse_move(stripped)
    except Exception:
        try:
            return normalize_move(stripped, stop_at_pass=stop_at_pass)
        except Exception:
            return "PASS"


def _extract_reward_move(
    completion: str,
    stop_at_pass: bool,
    profile: str,
) -> str:
    final_text = strip_reasoning_for_reward(completion, profile=profile)
    return _parse_completion_to_move(final_text, stop_at_pass=stop_at_pass)


def generate_model_move(
    model: object,
    tokenizer: object,
    board: BoardState,
    rollout_cfg: RolloutConfig,
    reasoning_profile: str,
    chat_template_kwargs: Optional[dict[str, object]] = None,
    generation_kwargs: Optional[dict[str, object]] = None,
) -> str:
    """Generate one model move from the current board prompt."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required at runtime
        logger.debug("Torch is unavailable during move generation; defaulting to PASS")
        return "PASS"

    prompt = build_prompt(board)
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
        "pad_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = int(inputs["input_ids"].shape[-1])
    completion_tokens = output_ids[0][prompt_len:]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    return _extract_reward_move(
        completion=completion_text,
        stop_at_pass=rollout_cfg.stop_at_pass,
        profile=reasoning_profile,
    )


def _apply_move_or_penalize(
    board: BoardState,
    player: str,
    move: str,
    reward_cfg: RewardConfig,
) -> tuple[bool, float, float, bool]:
    """Apply move if legal and return (applied, legality, length, ended_by_illegal)."""
    legal = is_legal_move(board, player, move, reward_cfg)
    legality = legality_reward(legal, reward_cfg)
    length = length_reward(move, reward_cfg)
    if not legal:
        return False, legality, length, reward_cfg.illegal_move_ends_game
    strict_apply = resolve_legality_level(reward_cfg) != LegalityLevel.BASIC
    board.apply_move(player, move, strict=strict_apply)
    return True, legality, length, False


def simulate_full_game(
    board: BoardState,
    initial_completion: str,
    model: object,
    tokenizer: object,
    katago: _KataGoService,
    reward_cfg: RewardConfig,
    rollout_cfg: RolloutConfig,
    fullgame_cfg: FullGameConfig,
    model_color: str,
    reasoning_cfg: Optional[object] = None,
    reasoning_model_name: str = "",
    analysis_cache: Optional[MutableMapping[tuple, KataGoResult]] = None,
    max_cache_size: Optional[int] = None,
) -> FullGameEpisodeResult:
    """Run one interactive game continuation and return a scalar reward breakdown."""
    model_color = model_color.upper()
    if board.next_player() != model_color:
        raise ValueError(
            f"Board to-move ({board.next_player()}) does not match model_color ({model_color})."
        )
    opponent_color = "W" if model_color == "B" else "B"
    if reasoning_cfg is None:
        from .config import ReasoningConfig

        reasoning_cfg = ReasoningConfig(enabled=False)
    reasoning_profile = detect_reasoning_profile(
        model_name=reasoning_model_name,
        tokenizer=tokenizer,
        configured_profile=getattr(reasoning_cfg, "profile", "plain"),
    )
    generation_kwargs = build_generation_kwargs(reasoning_cfg, reasoning_profile)
    chat_template_kwargs = build_chat_template_kwargs(reasoning_cfg, reasoning_profile)

    scorelead_delta_sum = 0.0
    legality_total = 0.0
    length_total = 0.0
    model_turns = 0
    plies_played = 0
    consecutive_passes = 0
    truncated = False
    terminated_by_illegal = False
    pass_terminated = False
    katago_fallback_passes = 0

    def _analyze_position(current_board: BoardState) -> KataGoResult:
        if analysis_cache is None:
            return katago.analyze_position(current_board)
        key = board_key(current_board)
        cached = analysis_cache.get(key)
        if cached is not None:
            mover = getattr(analysis_cache, "move_to_end", None)
            if callable(mover):
                mover(key)
            return cached
        result = katago.analyze_position(current_board)
        analysis_cache[key] = result
        if max_cache_size is not None and max_cache_size > 0:
            while len(analysis_cache) > max_cache_size:
                popitem = getattr(analysis_cache, "popitem", None)
                if callable(popitem):
                    try:
                        popitem(last=False)  # OrderedDict API (FIFO eviction)
                        continue
                    except TypeError:
                        pass
                # Generic MutableMapping fallback.
                first_key = next(iter(analysis_cache))
                del analysis_cache[first_key]
        return result

    def _record_pass(move_text: str) -> None:
        nonlocal consecutive_passes
        if move_text.upper() == "PASS":
            consecutive_passes += 1
        else:
            consecutive_passes = 0

    def _terminal_due_to_passes() -> bool:
        return fullgame_cfg.termination_two_passes and consecutive_passes >= 2

    def _play_model_turn(move_text: str) -> bool:
        nonlocal scorelead_delta_sum
        nonlocal legality_total
        nonlocal length_total
        nonlocal model_turns
        nonlocal plies_played
        nonlocal terminated_by_illegal

        pre = _analyze_position(board)
        pre_lead = scorelead_from_model_perspective(pre.score_lead, model_color)
        applied, legality, length, ended_illegal = _apply_move_or_penalize(
            board,
            model_color,
            move_text,
            reward_cfg,
        )
        legality_total += legality
        length_total += length
        model_turns += 1
        if ended_illegal:
            terminated_by_illegal = True
            return False

        if applied:
            post = _analyze_position(board)
            post_lead = scorelead_from_model_perspective(post.score_lead, model_color)
            scorelead_delta_sum += post_lead - pre_lead
        _record_pass(move_text)
        plies_played += 1
        return True

    first_move = _extract_reward_move(
        completion=initial_completion,
        stop_at_pass=rollout_cfg.stop_at_pass,
        profile=reasoning_profile,
    )
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
            opponent_move = katago.choose_move(board, player=opponent_color)
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
                generation_kwargs=generation_kwargs,
            )
            _play_model_turn(generated_move)

    if terminated_by_illegal:
        final_outcome = -1.0
    else:
        final_position = _analyze_position(board)
        final_lead = scorelead_from_model_perspective(final_position.score_lead, model_color)
        if final_lead > 0:
            final_outcome = 1.0
        elif final_lead < 0:
            final_outcome = -1.0
        else:
            final_outcome = 0.0

    scorelead_reward = reward_cfg.scorelead_scale * scorelead_delta_sum
    terminal_reward = reward_cfg.terminal_outcome_bonus * final_outcome
    total_reward = scorelead_reward + terminal_reward + legality_total + length_total
    return FullGameEpisodeResult(
        total_reward=total_reward,
        scorelead_reward=scorelead_reward,
        terminal_reward=terminal_reward,
        legality_reward_total=legality_total,
        length_reward_total=length_total,
        model_turns=model_turns,
        plies_played=plies_played,
        truncated=truncated,
        terminated_by_illegal=terminated_by_illegal,
        pass_terminated=pass_terminated,
        model_color=model_color,
        scorelead_delta_sum=scorelead_delta_sum,
        final_outcome=final_outcome,
        katago_fallback_passes=katago_fallback_passes,
    )
