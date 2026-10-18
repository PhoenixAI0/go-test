from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", ["run_full_pipeline.sh", "train_cuda.sh"])
def test_pipeline_dry_run_and_validation_without_setup(tmp_path, entrypoint):
    # Isolate from the checkout's .env, data and installed environment.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_full_pipeline.sh", "train_cuda.sh", "train_cuda.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    env = dict(os.environ)
    env.pop("MODEL_GTP_COMMAND_TEMPLATE", None)
    result = subprocess.run(
        ["bash", str(scripts / entrypoint), "--dry-run", "--skip-sft", "--skip-fullgame",
         "--model-command-template", "engine --model {MODEL_DIR}",
         "--upload-to-hub", "--hf-repo-id", "user/repo"],
        cwd="/", env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "grpo_9x9 -> grpo_19x19" in result.stdout
    assert "promotion gates: True; Hub upload: True" in result.stdout
    assert not (tmp_path / "outputs").exists()
    result = subprocess.run(
        ["bash", str(scripts / entrypoint), "--upload-to-hub"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--hf-repo-id is required" in result.stderr
    (tmp_path / ".env").write_text("MODEL_GTP_COMMAND_TEMPLATE='engine --model {MODEL_DIR}'\n")
    for flags, expected in (([], "True"), (["--no-eval"], "False")):
        result = subprocess.run(
            ["bash", str(scripts / entrypoint), "--dry-run", *flags],
            env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert f"promotion gates: {expected}" in result.stdout


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import go_rl.katago
    from go_rl.board import BoardState

    spec = importlib.util.spec_from_file_location("pipeline_launcher", ROOT / "scripts/train_cuda.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_GTP_COMMAND_TEMPLATE", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    shutil.copytree(ROOT / "config", tmp_path / "config")
    binary = tmp_path / "katago"
    binary.touch()
    for key in ("KATAGO_BIN", "KATAGO_MODEL", "KATAGO_CONFIG"):
        monkeypatch.setenv(key, str(binary))
    monkeypatch.setattr(launcher, "cuda_info", lambda *args: {"gpu": "test GPU"})
    monkeypatch.setattr(go_rl.katago, "KataGoAnalysisClient", lambda cfg: SimpleNamespace(
        start=lambda: None, close=lambda: None,
        analyze_position=lambda board: SimpleNamespace(winrate=0.5)))
    data = tmp_path / "prepared"
    data.mkdir()
    for size in (9, 19):
        row = json.dumps({"prompt": BoardState.empty(size).to_prompt(), "response": "D4"})
        for prefix in ("sft", "rl_prompts"):
            (data / f"{prefix}_{size}x{size}.jsonl").write_text(row + "\n")
    state = SimpleNamespace(launcher=launcher, calls=[], fail_gate=None,
                            fail_eval=False, missing_adapter=False, run_dir=tmp_path / "run")

    def fake_run(command):
        state.calls.append(command)
        script = Path(command[1]).name
        if script in ("run_sft.py", "run_grpo.py", "run_grpo_fullgame.py"):
            config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            output = Path(config.get("training", config)["output_dir"])
            output.mkdir()
            if not state.missing_adapter:
                for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
                    (output / name).write_text("{}")
        elif script == "eval_elo.py":
            if state.fail_eval:
                raise subprocess.CalledProcessError(1, command)
            path = Path(command[command.index("--summary-json") + 1])
            failed = path.name == f"{state.fail_gate}_summary.json"
            path.write_text(json.dumps({"elo_delta": -50 if failed else 200,
                                        "win_rate": 0.1 if failed else 0.7}))
        elif script == "create_match_config.py":
            # Exercise the actual CLI contract without launching a match or GPU work.
            result = subprocess.run([command[0], str(ROOT / command[1]), *command[2:]],
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
        else:
            assert script == "upload_to_hub.py"

    monkeypatch.setattr(launcher, "run", fake_run)

    def start(*flags):
        monkeypatch.setattr(sys, "argv", ["pipeline", "--skip-setup", "--skip-data",
                            "--data-dir", str(data), "--output-dir", str(state.run_dir), *flags])
        launcher.main()

    state.start = start
    return state


@pytest.mark.parametrize("gate_failure", [None, "grpo_9x9", "grpo_19x19"])
def test_pipeline_evaluates_gates_and_uploads_final_adapter(pipeline, gate_failure):
    pipeline.fail_gate = gate_failure
    flags = ["--smoke", "--model-command-template", 'engine --model "{MODEL_DIR}" --board {BOARD_SIZE}',
             "--upload-to-hub", "--hf-repo-id", "user/repo", "--hf-private",
             "--wandb", "--wandb-project", "test-project", "--fullgame-smoke-plies", "6"]
    if gate_failure == "grpo_9x9":
        with pytest.raises(RuntimeError, match="9x9 promotion gate failed"):
            pipeline.start(*flags)
    else:
        pipeline.start(*flags)
    scripts = [Path(c[1]).name for c in pipeline.calls]
    expected = ["run_sft.py", "create_match_config.py", "eval_elo.py",
                "run_grpo.py", "create_match_config.py", "eval_elo.py"]
    if gate_failure != "grpo_9x9":
        expected += ["run_grpo.py", "create_match_config.py", "eval_elo.py"]
        if gate_failure is None:
            expected += ["run_grpo_fullgame.py", "create_match_config.py", "eval_elo.py"]
        expected += ["upload_to_hub.py"]
    assert scripts == expected
    for command in pipeline.calls:
        if Path(command[1]).name == "create_match_config.py":
            assert command[command.index("--num-games") + 1] == "8"
            model_command = command[command.index("--model-command") + 1]
            assert str(pipeline.run_dir) in model_command
            assert "{MODEL_DIR}" not in model_command and "{BOARD_SIZE}" not in model_command
        if "--wandb" in command:
            assert command[command.index("--wandb-project") + 1] == "test-project"
    if gate_failure != "grpo_9x9":
        upload = pipeline.calls[-1]
        final_stage = "grpo_19x19" if gate_failure else "grpo_fullgame_19x19"
        assert upload[upload.index("--model") + 1] == str(pipeline.run_dir / final_stage)
        assert "--private" in upload and "--push-only" in upload
    if gate_failure is None:
        fullgame = next(c for c in pipeline.calls if c[1].endswith("run_grpo_fullgame.py"))
        assert fullgame[fullgame.index("--max-plies") + 1] == "6"


@pytest.mark.parametrize("skip_fullgame", [False, True])
def test_pipeline_no_eval_bypasses_gates_and_reuses_model(pipeline, skip_fullgame):
    flags = ["--no-eval", "--model-command-template", "ignored", "--skip-sft", "--model", "existing-adapter"]
    if skip_fullgame:
        flags.append("--skip-fullgame")
    pipeline.start(*flags)
    scripts = [Path(c[1]).name for c in pipeline.calls]
    assert scripts == ["run_grpo.py", "run_grpo.py"] + ([] if skip_fullgame else ["run_grpo_fullgame.py"])
    config = yaml.safe_load((pipeline.run_dir / "config/grpo_9x9.yaml").read_text())
    assert config["training"]["model_name"] == "existing-adapter"


@pytest.mark.parametrize("failure", ["eval", "adapter"])
def test_pipeline_stops_before_upload_on_failure(pipeline, failure):
    pipeline.fail_eval = failure == "eval"
    pipeline.missing_adapter = failure == "adapter"
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        pipeline.start("--model-command-template", "engine {MODEL_DIR}",
                       "--upload-to-hub", "--hf-repo-id", "user/repo")
    assert not any(c[1].endswith("upload_to_hub.py") for c in pipeline.calls)
    assert not any(c[1].endswith("run_grpo.py") for c in pipeline.calls)
