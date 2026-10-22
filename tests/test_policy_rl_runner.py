import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("policy_rl_runner_test", Path(__file__).resolve().parents[1] / "scripts/run_policy_rl.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_fresh_run_rejects_existing_experiment_without_truncating_logs(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    metrics = output / "metrics.jsonl"
    metrics.write_text('{"collected_moves": 100}\n')
    with pytest.raises(ValueError, match="nonempty"):
        runner.validate_run_destination(str(tmp_path / "seed"), str(output))
    assert metrics.read_text() == '{"collected_moves": 100}\n'
    runner.validate_run_destination(str(tmp_path / "seed"), str(tmp_path / "new"))
    with pytest.raises(ValueError, match="starting checkpoint"):
        runner.validate_run_destination(str(output), str(output))


@pytest.mark.parametrize("readout", ["cells", "cell_attention"])
def test_cell_architecture_cannot_be_silently_given_text_encoding(readout):
    metadata = {"readout": readout, "encoding": "cells"}
    assert runner.resolve_run_encoding(metadata, None) == "cells"
    assert runner.resolve_run_encoding(metadata, "cells") == "cells"
    with pytest.raises(ValueError, match="require cells encoding"):
        runner.resolve_run_encoding(metadata, "compact")
    with pytest.raises(ValueError, match="require cells encoding"):
        runner.resolve_run_encoding({"readout": readout}, None)


def test_last_token_readout_retains_explicit_encoding_ablation():
    assert runner.resolve_run_encoding({"encoding": "compact", "readout": "last"}, "grid") == "grid"
