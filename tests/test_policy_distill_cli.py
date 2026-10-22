import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "run_policy_distill", Path(__file__).resolve().parents[1] / "scripts" / "run_policy_distill.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_restored_checkpoint_does_not_silently_ignore_head_only(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "run_policy_distill.py", "--train-data", "unused-train", "--eval-data", "unused-eval",
        "--output-dir", "unused-output", "--init-checkpoint", "unused-checkpoint", "--head-only",
    ])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
