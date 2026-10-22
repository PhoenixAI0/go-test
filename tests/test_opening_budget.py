import importlib.util
import json
from pathlib import Path

import pytest

from go_rl.rules import normalize_rules
from go_rl.teacher_data import TARGET_RULES

spec = importlib.util.spec_from_file_location(
    "prepare_opening_budget_test", Path(__file__).resolve().parents[1] / "scripts" / "prepare_opening_budget.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.mark.parametrize("split", ["train", "", None, "unknown"])
def test_opening_exclusions_require_explicit_heldout_roles(split):
    with pytest.raises(ValueError, match="pinned to validation or test"):
        preparation.validate_exclusion_objectives([{"rules": TARGET_RULES, "komi": 7.5, "split": split}])


def test_opening_exclusions_use_the_same_normalized_target_identity():
    row = {"split": "validation", "rules": TARGET_RULES, "komi": 7.5}
    preparation.validate_exclusion_objectives([row, {**row, "split": "test", "rules": normalize_rules(TARGET_RULES)}])
    for changed in ({**row, "rules": "chinese"}, {**row, "komi": 6.5}, {**row, "komi": float("nan")}):
        with pytest.raises(ValueError, match="target rules and komi"):
            preparation.validate_exclusion_objectives([changed])


def test_opening_cli_rejects_raw_holdout_objective_before_source_selection(tmp_path, monkeypatch):
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"split": "test", "rules": "japanese", "komi": 6.5}) + "\n")
    output = tmp_path / "new-source.jsonl"
    monkeypatch.setattr("sys.argv", [
        "prepare_opening_budget.py", "--dataset", "must-not-be-opened", "--background", "must-not-be-opened",
        "--exclude-dataset", str(heldout), "--output", str(output),
    ])
    with pytest.raises(ValueError, match="target rules and komi"):
        preparation.main()
    assert not output.exists() and not output.with_suffix(".manifest.json").exists()
