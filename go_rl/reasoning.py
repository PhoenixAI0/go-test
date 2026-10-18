from __future__ import annotations

import logging
import re
from dataclasses import asdict, is_dataclass
from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

_QWEN_THINK_START = re.compile(r"<think>", re.IGNORECASE)
_QWEN_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_QWEN_THINK_END = re.compile(r"</think>", re.IGNORECASE)


def _keyword_profiles(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    if "gpt-oss" in lowered or "gpt_oss" in lowered:
        matches.append("gpt_oss")
    if "glm" in lowered or "z.ai" in lowered or "zai" in lowered:
        matches.append("glm47")
    if "qwen" in lowered:
        matches.append("qwen3")
    return matches


def detect_reasoning_profile(
    model_name: str,
    tokenizer: object,
    configured_profile: str,
) -> str:
    allowed = {"auto", "qwen3", "gpt_oss", "glm47", "plain"}
    if configured_profile not in allowed:
        raise ValueError(f"Unknown reasoning profile: {configured_profile}")
    if configured_profile != "auto":
        return configured_profile

    model_text = str(model_name or "")
    # Saved adapters use local directory names without a model-family marker.
    tokenizer_text = " ".join([
        str(getattr(tokenizer, "name_or_path", "") or ""),
        type(tokenizer).__name__,
    ])
    model_matches = _keyword_profiles(model_text)
    tokenizer_matches = _keyword_profiles(tokenizer_text)

    if len(model_matches) > 1:
        logger.warning(
            "Ambiguous reasoning profile markers in model name '%s' (%s); using default '%s'.",
            model_text,
            ", ".join(model_matches),
            model_matches[0],
        )
    if len(tokenizer_matches) > 1:
        logger.warning(
            "Ambiguous reasoning profile markers in tokenizer name '%s' (%s); using default '%s'.",
            tokenizer_text,
            ", ".join(tokenizer_matches),
            tokenizer_matches[0],
        )

    if model_matches:
        if tokenizer_matches and tokenizer_matches[0] != model_matches[0]:
            logger.warning(
                "Model/tokenizer reasoning profile mismatch (model='%s', tokenizer='%s'); using model default '%s'.",
                model_matches[0],
                tokenizer_matches[0],
                model_matches[0],
            )
        return model_matches[0]

    if tokenizer_matches:
        return tokenizer_matches[0]

    logger.warning(
        "Unable to auto-detect reasoning profile for model '%s'; falling back to plain.",
        model_name,
    )
    return "plain"


def _to_plain_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        return _to_plain_text(asdict(value))
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in ("final", "content", "text", "output_text", "message"):
            if key in value:
                parts.append(_to_plain_text(value[key]))
        if not parts:
            parts = [_to_plain_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_to_plain_text(item) for item in value).strip()
    return str(value)


def _extract_qwen3(completion: object) -> tuple[str, bool, bool]:
    text = _to_plain_text(completion)
    start_count = len(_QWEN_THINK_START.findall(text))
    end_count = len(_QWEN_THINK_END.findall(text))
    has_reasoning = start_count > 0

    if has_reasoning and end_count < start_count:
        return "", True, True

    # Qwen's generation prompt can already contain the opening <think> tag.
    if not has_reasoning and end_count:
        return _QWEN_THINK_END.split(text)[-1].strip(), True, False

    if not has_reasoning:
        return text.strip(), False, False

    without_thinking = _QWEN_THINK_BLOCK.sub("", text).strip()
    return without_thinking, True, False


def _collect_channel_payload(entry: object) -> tuple[str | None, str]:
    if isinstance(entry, Mapping):
        channel = entry.get("channel") or entry.get("type") or entry.get("role")
        payload = entry.get("content")
        if payload is None:
            payload = entry.get("text")
        if payload is None and "message" in entry:
            payload = entry["message"]
        return (str(channel).lower() if channel else None, _to_plain_text(payload))
    return (None, _to_plain_text(entry))


def _extract_gpt_oss(completion: object) -> tuple[str, bool, bool]:
    data: Iterable[object]
    if isinstance(completion, Mapping):
        if isinstance(completion.get("output"), list):
            data = completion["output"]
        elif isinstance(completion.get("messages"), list):
            data = completion["messages"]
        else:
            data = [completion]
    elif isinstance(completion, Sequence) and not isinstance(completion, (str, bytes, bytearray)):
        data = completion
    else:
        text = _to_plain_text(completion).strip()
        return text, False, False

    final_parts: list[str] = []
    analysis_parts: list[str] = []
    fallback_parts: list[str] = []
    for entry in data:
        channel, payload = _collect_channel_payload(entry)
        if not payload:
            continue
        if channel in {"final", "assistant_final"}:
            final_parts.append(payload)
        elif channel in {"analysis", "reasoning"}:
            analysis_parts.append(payload)
        else:
            fallback_parts.append(payload)

    has_reasoning = len(analysis_parts) > 0
    if final_parts:
        return "\n".join(final_parts).strip(), has_reasoning, False
    if has_reasoning:
        return "", True, True
    return "\n".join(fallback_parts).strip(), False, False


def _extract_glm47(completion: object) -> tuple[str, bool, bool]:
    if isinstance(completion, Mapping):
        final_text = _to_plain_text(completion.get("content")).strip()
        reasoning_text = _to_plain_text(completion.get("reasoning_content")).strip()
        has_reasoning = bool(reasoning_text)
        if has_reasoning and not final_text:
            return "", True, True
        if final_text:
            return final_text, has_reasoning, False
    text = _to_plain_text(completion).strip()
    return text, False, False


def extract_final_answer(completion: object, profile: str) -> tuple[str, bool, bool]:
    if profile == "qwen3":
        return _extract_qwen3(completion)
    if profile == "gpt_oss":
        return _extract_gpt_oss(completion)
    if profile == "glm47":
        return _extract_glm47(completion)
    if profile == "plain":
        return _to_plain_text(completion).strip(), False, False
    raise ValueError(f"Unknown reasoning profile: {profile}")


def strip_reasoning_for_reward(completion: object, profile: str) -> str:
    final_text, _has_reasoning, malformed = extract_final_answer(completion, profile)
    if malformed:
        return ""
    return final_text.strip()


def apply_malformed_policy(length: int, malformed_policy: str) -> list[int]:
    if malformed_policy != "mask_all":
        raise ValueError(f"Unsupported malformed_policy: {malformed_policy}")
    return [0] * max(0, length)


def _token_char_spans(completion_ids: Sequence[int], tokenizer: object) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    previous = ""
    for index in range(len(completion_ids)):
        current = tokenizer.decode(completion_ids[: index + 1], skip_special_tokens=True)
        spans.append((len(previous), len(current)))
        previous = current
    return spans


def _build_mask_from_char_span(
    completion_ids: Sequence[int],
    tokenizer: object,
    start: int,
    end: int,
) -> list[int]:
    if not completion_ids:
        return []
    spans = _token_char_spans(completion_ids, tokenizer)
    mask: list[int] = []
    for token_start, token_end in spans:
        overlaps = token_start < end and token_end > start
        mask.append(1 if overlaps else 0)
    return mask


def build_final_only_mask(
    completion: object,
    completion_ids: Sequence[int],
    tokenizer: object,
    profile: str,
    malformed_policy: str = "mask_all",
) -> tuple[list[int], bool, int]:
    if not completion_ids:
        return [], False, 0

    completion_text = _to_plain_text(completion)
    final_text, has_reasoning, malformed = extract_final_answer(completion, profile)

    if malformed:
        return apply_malformed_policy(len(completion_ids), malformed_policy), True, 0

    if not has_reasoning:
        return [1] * len(completion_ids), False, len(completion_ids)

    stripped_final = final_text.strip()
    if not stripped_final:
        return [0] * len(completion_ids), False, 0

    start = completion_text.rfind(stripped_final)
    if start < 0:
        start = completion_text.strip().rfind(stripped_final)
    if start < 0:
        return apply_malformed_policy(len(completion_ids), malformed_policy), True, 0

    end = start + len(stripped_final)
    mask = _build_mask_from_char_span(completion_ids, tokenizer, start, end)
    final_len = sum(mask)
    if final_len == 0:
        return apply_malformed_policy(len(completion_ids), malformed_policy), True, 0
    return mask, False, final_len


def build_chat_template_kwargs(reasoning_cfg: object, profile: str) -> dict[str, object]:
    if not getattr(reasoning_cfg, "enabled", True):
        return {}
    if profile == "qwen3":
        return {"enable_thinking": bool(getattr(reasoning_cfg, "qwen_enable_thinking", True))}
    if profile == "gpt_oss":
        return {"reasoning_effort": str(getattr(reasoning_cfg, "gpt_oss_reasoning_effort", "medium"))}
    if profile == "glm47":
        return {
            "thinking_type": str(getattr(reasoning_cfg, "glm_thinking_type", "enabled")),
            "clear_thinking": bool(getattr(reasoning_cfg, "glm_clear_thinking", False)),
        }
    return {}
