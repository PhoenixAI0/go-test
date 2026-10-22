import pytest

from go_rl.rules import UnsupportedRulesError, legality_superko, normalize_rules


def test_chinese_shorthand_is_simple_not_positional_ko():
    assert not legality_superko("chinese")
    assert legality_superko("chinese-ogs")
    assert normalize_rules("Chinese KGS") == normalize_rules("chinese-ogs")


def test_compact_rules_preserve_all_nondefault_scoring_details():
    raw = "koPOSITIONALscoreAREAtaxSEKIsui0button1whbN-1fpok1"
    normalized = normalize_rules(raw)
    assert normalized == {
        "ko": "POSITIONAL", "scoring": "AREA", "tax": "SEKI", "suicide": False,
        "hasButton": True, "whiteHandicapBonus": "N-1", "friendlyPassOk": True,
    }
    assert normalize_rules(normalized) == normalized


def test_raw_rule_defaults_are_not_chinese_server_defaults():
    raw = normalize_rules("koPOSITIONALscoreAREAtaxNONEsui0")
    named = normalize_rules("chinese-ogs")
    assert raw["whiteHandicapBonus"] == "0" and not raw["friendlyPassOk"]
    assert named["whiteHandicapBonus"] == "N" and named["friendlyPassOk"]


def test_json_defaults_match_katago_and_do_not_assume_no_suicide():
    rules = normalize_rules('{"ko":"SIMPLE","scoring":"TERRITORY","suicide":false}')
    assert rules["tax"] == "SEKI" and not rules["friendlyPassOk"]
    assert not legality_superko(rules)
    with pytest.raises(UnsupportedRulesError, match="suicide"):
        legality_superko({"ko": "POSITIONAL"})


@pytest.mark.parametrize("raw", ["aga", "new-zealand", "tromp-taylor", "koSPIGHTscoreAREAtaxNONEsui0"])
def test_unsupported_local_legality_is_explicit(raw):
    normalize_rules(raw)  # The engine may support it, but our local mask cannot.
    with pytest.raises(UnsupportedRulesError):
        legality_superko(raw)


@pytest.mark.parametrize("raw", ["", "unknown", "koPOSITIONALscoreAREAtaxNONEsui0garbage",
                                  {"suicide": "false"}, {"komi": 7.5},
                                  {"ko": "TYPO"}, {"scoring": "TERRITORY", "hasButton": True}])
def test_invalid_rules_never_fall_back_to_other_rules(raw):
    with pytest.raises(ValueError):
        normalize_rules(raw)
