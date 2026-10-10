from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location("train_cuda", ROOT / "scripts/train_cuda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_dry_run_without_cuda_or_data(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "scripts/train_cuda.sh"), "--dry-run"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "sft_9x9 -> grpo_9x9 -> grpo_19x19 -> grpo_fullgame_19x19" in result.stdout


def test_launcher_rejects_missing_data_before_setup(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "scripts/train_cuda.sh"), "--sgf-9x9", str(tmp_path / "missing")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "No SGF games" in result.stderr
    assert "Downloading" not in result.stdout


@pytest.mark.parametrize("fail_stage", [None, "grpo_9x9"])
def test_launcher_handoffs_and_stops_on_failure(tmp_path, monkeypatch, fail_stage):
    from go_rl.board import BoardState
    import go_rl.katago

    launcher = load_launcher()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    shutil.copytree(ROOT / "config", tmp_path / "config")
    binary = tmp_path / "katago"
    binary.touch()
    monkeypatch.setenv("KATAGO_BIN", str(binary))
    monkeypatch.setenv("KATAGO_MODEL", str(binary))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(launcher, "cuda_info", lambda *args: {
        "torch": "test", "cuda": "test", "gpu": "test CUDA device"})
    monkeypatch.setattr(go_rl.katago, "KataGoAnalysisClient", lambda cfg: SimpleNamespace(
        start=lambda: None, close=lambda: None,
        analyze_position=lambda board: SimpleNamespace(winrate=0.5)))
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_cuda", "--smoke", "--output-dir", str(run_dir)])
    calls = []

    def fake_run(command):
        if command[1].endswith("prepare_datasets.py"):
            tag = command[command.index("--tag") + 1]
            data_dir = Path(command[command.index("--output-dir") + 1])
            size = int(tag.split("x")[0])
            row = {"prompt": BoardState.empty(size).to_prompt(), "response": "D4"}
            (data_dir / f"sft_{tag}.jsonl").write_text(json.dumps(row) + "\n")
            return
        config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        training = config.get("training", config)
        output = Path(training["output_dir"])
        calls.append((output.name, training["model_name"]))
        if output.name == fail_stage:
            raise subprocess.CalledProcessError(7, command)
        output.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
            (output / name).write_text("{}")
        assert "--smoke" in command
        assert "--gradient-checkpointing" in command

    monkeypatch.setattr(launcher, "run", fake_run)
    if fail_stage:
        with pytest.raises(subprocess.CalledProcessError):
            launcher.main()
        assert [name for name, _ in calls] == ["sft_9x9", "grpo_9x9"]
    else:
        launcher.main()
        assert calls == [
            ("sft_9x9", "Qwen/Qwen3-8B"),
            ("grpo_9x9", str(run_dir / "sft_9x9")),
            ("grpo_19x19", str(run_dir / "grpo_9x9")),
            ("grpo_fullgame_19x19", str(run_dir / "grpo_19x19")),
        ]
