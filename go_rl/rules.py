"""Exact KataGo rule metadata and the subset supported by local legality.

See KataGo cpp/game/rules.cpp (v1.18.2). In particular, ``chinese`` means
simple ko; only ``chinese-ogs``/``chinese-kgs`` use positional superko.
Komi is kept separate from this mapping and must accompany teacher requests.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

_COMPACT = re.compile(
    r"ko(SIMPLE|POSITIONAL|SITUATIONAL|SPIGHT)score(AREA|TERRITORY)"
    r"tax(NONE|SEKI|ALL)sui([01])(?:button([01]))?"
    r"(?:whb(N-1|N|0))?(?:fpok([01]))?",
    re.IGNORECASE,
)
_DEFAULTS: dict[str, object] = {
    "ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE",
    "suicide": True, "hasButton": False,
    "whiteHandicapBonus": "0", "friendlyPassOk": False,
}


class UnsupportedRulesError(ValueError):
    """The local Go implementation cannot faithfully enforce these rules."""


def normalize_rules(rules: str | Mapping[str, object]) -> dict[str, object]:
    """Expand a supported name, KataGo compact RU, or JSON rule mapping.

    This preserves scoring/tax/button/handicap semantics. It does not silently
    equate a raw SGF rule string with a named ruleset or assume missing SGF rules.
    Normalizing metadata does not imply that local legality supports that rule.
    """
    if isinstance(rules, Mapping):
        values = dict(rules)
        if "score" in values:
            if "scoring" in values and values["scoring"] != values["score"]:
                raise ValueError("Conflicting score and scoring rules")
            values["scoring"] = values.pop("score")
        unknown = set(values) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rules fields: {sorted(unknown)}; pass komi separately")
        result = {**_DEFAULTS, **values}
        if "tax" not in values and result["scoring"] == "TERRITORY":
            result["tax"] = "SEKI"
    elif isinstance(rules, str) and rules.strip():
        raw = rules.strip()
        if raw.startswith("{"):
            decoded = json.loads(raw)
            if not isinstance(decoded, dict):
                raise ValueError("Rules JSON must be an object")
            return normalize_rules(decoded)
        match = _COMPACT.fullmatch(raw)
        if match:
            ko, scoring, tax, suicide, button, bonus, friendly = match.groups()
            result = dict(_DEFAULTS)
            result.update(ko=ko.upper(), scoring=scoring.upper(), tax=tax.upper(),
                          suicide=suicide == "1", hasButton=button == "1",
                          whiteHandicapBonus=(bonus or "0").upper(), friendlyPassOk=friendly == "1")
        else:
            name = re.sub(r"[-_\s]", "", raw.lower())
            result = dict(_DEFAULTS)
            if name in {"japanese", "korean"}:
                result.update(ko="SIMPLE", scoring="TERRITORY", tax="SEKI", suicide=False)
            elif name == "chinese":
                result.update(ko="SIMPLE", suicide=False, whiteHandicapBonus="N", friendlyPassOk=True)
            elif name in {"chineseogs", "chinesekgs"}:
                result.update(suicide=False, whiteHandicapBonus="N", friendlyPassOk=True)
            elif name in {"stonescoring", "ancientarea"}:
                result.update(ko="SIMPLE", tax="ALL", suicide=False, friendlyPassOk=True)
            elif name == "ancientterritory":
                result.update(ko="SIMPLE", scoring="TERRITORY", tax="ALL", suicide=False)
            elif name in {"aga", "bga", "french", "agabutton"}:
                result.update(ko="SITUATIONAL", suicide=False, whiteHandicapBonus="N-1",
                              friendlyPassOk=True, hasButton=name == "agabutton")
            elif name in {"nz", "newzealand"}:
                result.update(ko="SITUATIONAL", friendlyPassOk=True)
            elif name in {"goe", "ing"}:
                result.update(friendlyPassOk=True)
            elif name != "tromptaylor":
                raise ValueError(f"Unknown Go rules: {rules!r}")
    else:
        raise ValueError("Rules must be a nonempty name or mapping; missing SGF rules need an explicit default")

    for key, choices in (("ko", {"SIMPLE", "POSITIONAL", "SITUATIONAL", "SPIGHT"}),
                         ("scoring", {"AREA", "TERRITORY"}),
                         ("tax", {"NONE", "SEKI", "ALL"}),
                         ("whiteHandicapBonus", {"0", "N", "N-1"})):
        value = result[key]
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Invalid {key} rule: {value!r}")
    for key in ("suicide", "hasButton", "friendlyPassOk"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if result["hasButton"] and result["scoring"] != "AREA":
        raise ValueError("Button Go requires area scoring")
    return result


def legality_superko(rules: str | Mapping[str, object]) -> bool:
    """Return the local positional-superko flag, failing on unsupported rules.

    This is main-phase move legality only. Territory-scoring encore phases and
    final scores belong to the trusted engine, never the local board renderer.
    """
    normalized = normalize_rules(rules)
    if normalized["suicide"]:
        raise UnsupportedRulesError("Local legality does not support multi-stone suicide")
    if normalized["ko"] not in {"SIMPLE", "POSITIONAL"}:
        raise UnsupportedRulesError(f"Local legality does not support {normalized['ko']} ko")
    return normalized["ko"] == "POSITIONAL"
