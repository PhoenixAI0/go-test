"""Recipe contracts: cheap dry runs, isolated inputs, and auditable subprocesses."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from go_rl.rules import normalize_rules

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_19x19", ROOT / "scripts/train_19x19.py")
recipe_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe_module)


def arguments(tmp_path, **changes):
    values = dict(config=str(ROOT / "config/policy_19x19_qwen06.yaml"),
                  dataset=None, train_data=None, eval_data=None,
                  output_dir=str(tmp_path / "run"), device="cpu", selfplay=False,
                  evaluate=False, smoke=False, dry_run=False, train_kind=None, encoding=None,
                  resume_from=None, init_checkpoint=None)
    return argparse.Namespace(**(values | changes))


def write_records(path, prefix, count, split):
    rows = [{"board_size": 19, "game_id": f"{prefix}-{index}", "move_number": index + 10,
             "prompt": f"position-{prefix}-{index}", "split": split,
             "rules": "koPOSITIONALscoreAREAtaxNONEsui0", "komi": 7.5,
             "response": "A1", "policy_target": [1.] + [0.] * 361}
            for index in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def reuse_recipe(tmp_path, **changes):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    write_records(train, "train", 12, "train")
    write_records(validation, "validation", 6, "validation")
    return recipe_module.load_recipe(arguments(tmp_path, train_data=str(train),
                                               eval_data=str(validation), **changes))


def options(stage):
    return dict(zip(stage["argv"][2::2], stage["argv"][3::2], strict=True))


def test_dry_run_needs_no_engines_models_or_output_writes(tmp_path):
    destination = tmp_path / "does-not-exist"
    environment = {**os.environ, "KATAGO_BIN": "absent-katago", "KATAGO_MODEL": "absent-model",
                   "HF_HUB_OFFLINE": "1"}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/train_19x19.py"),
                             "--dry-run", "--smoke", "--evaluate", "--selfplay",
                             "--train-kind", "categorical",
                             "--output-dir", str(destination)],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not destination.exists()
    assert "[teacher_train]" in result.stdout
    assert "[teacher_validation]" in result.stdout
    assert "[distill]" in result.stdout
    assert "[selfplay]" in result.stdout
    assert "[move_quality]" in result.stdout
    assert "[matches]" in result.stdout
    assert "c1899de289a04d12100db370d81485cdf75e47ca" in result.stdout


def test_stage_commands_preserve_rules_and_handoff_checkpoints(tmp_path):
    recipe = recipe_module.load_recipe(arguments(tmp_path, selfplay=True, evaluate=True, train_kind="categorical"))
    stages, match = recipe_module.build_plan(recipe)
    assert [stage["name"] for stage in stages] == [
        "teacher_train", "teacher_validation", "distill", "selfplay", "move_quality", "matches"]
    teacher, validation, distill, selfplay, quality, _ = map(options, stages)
    assert teacher["--split"] == "train" and validation["--split"] == "validation"
    assert teacher["--sampling"] == "game-phase" and "--max-move" not in teacher
    assert normalize_rules(teacher["--relabel-rules"]) == normalize_rules(recipe["rules"])
    assert distill["--encoding"] == distill["--readout"] == "cells"
    assert distill["--attention"] == "bidirectional"
    assert selfplay["--model"] == str(tmp_path / "run/distill/best")
    assert quality["--policy-checkpoint"] == "policy=" + str(tmp_path / "run/selfplay")
    model_command = shlex.split(match["model"]["command"])
    assert model_command[model_command.index("--checkpoint") + 1] == str(tmp_path / "run/selfplay")
    assert "--num-beams" not in model_command
    assert "--text-beam-widths" not in quality
    assert match["max_moves"] == 1083


def test_default_teacher_window_includes_empty_board_and_opening_states(tmp_path):
    recipe = recipe_module.load_recipe(arguments(tmp_path))
    assert recipe["teacher"]["min_move"] == 0
    stages, _ = recipe_module.build_plan(recipe)
    assert options(stages[0])["--min-move"] == "0"
    assert options(stages[1])["--min-move"] == "0"


def test_text_recipe_uses_text_training_evaluation_and_gtp(tmp_path):
    recipe = reuse_recipe(tmp_path, evaluate=True, encoding="grid")
    stages, match = recipe_module.build_plan(recipe)
    assert [stage["name"] for stage in stages] == ["distill", "move_quality", "matches"]
    assert stages[0]["argv"][1].endswith("run_text_distill.py")
    assert options(stages[0])["--encoding"] == "grid"
    assert "--readout" not in options(stages[0])
    assert "--text-model" in options(stages[1])
    assert "--policy-checkpoint" not in options(stages[1])
    command = shlex.split(match["model"]["command"])
    assert "--model" in command and "--checkpoint" not in command
    assert command[command.index("--encoding") + 1] == "grid"
    assert options(stages[1])["--text-beam-widths"] == "1"
    assert command[command.index("--num-beams") + 1] == "1"


def test_pilot_uses_frozen_labels_selected_training_and_fixed_text_beam(tmp_path):
    recipe = recipe_module.load_recipe(arguments(
        tmp_path, config=str(ROOT / "config/policy_19x19_qwen06_pilot.yaml"), evaluate=True))
    stages, match = recipe_module.build_plan(recipe)
    assert [stage["name"] for stage in stages] == ["distill", "move_quality", "matches"]
    training, quality = options(stages[0]), options(stages[1])
    assert training["--train-data"] == str(ROOT / "data/teacher_v4_visits64.jsonl")
    assert training["--eval-data"] == str(ROOT / "data/teacher_19x19_val_v3.jsonl")
    assert (training["--steps"], training["--batch-size"], training["--gradient-accumulation"]) == ("120", "4", "1")
    assert training["--eval-every"] == "60" and training["--symmetries"] == "8"
    assert training["--encoding"] == "compact" and float(training["--learning-rate"]) == 1e-4
    assert quality["--text-beam-widths"] == "4"
    model_command = shlex.split(match["model"]["command"])
    assert model_command[model_command.index("--num-beams") + 1] == "4"
    assert "--num-beams" not in training and "--text-beam-widths" not in training


def test_missing_beam_setting_preserves_greedy_default(tmp_path):
    raw = yaml.safe_load((ROOT / "config/policy_19x19_qwen06_pilot.yaml").read_text())
    raw["evaluation"].pop("text_num_beams")
    path = tmp_path / "old_recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    recipe = recipe_module.load_recipe(arguments(tmp_path, config=str(path), evaluate=True))
    stages, match = recipe_module.build_plan(recipe)
    assert recipe["evaluation"]["text_num_beams"] == 1
    assert options(stages[1])["--text-beam-widths"] == "1"
    command = shlex.split(match["model"]["command"])
    assert command[command.index("--num-beams") + 1] == "1"


def test_text_model_cannot_silently_enter_categorical_ppo(tmp_path):
    with pytest.raises(ValueError, match="cannot train a text model"):
        recipe_module.load_recipe(arguments(tmp_path, selfplay=True))
    assert not (tmp_path / "run").exists()


def test_resume_requires_reused_data_and_forwards_checkpoint_state(tmp_path):
    with pytest.raises(ValueError, match="explicit existing"):
        recipe_module.load_recipe(arguments(tmp_path, resume_from="prior-checkpoint"))
    recipe = reuse_recipe(tmp_path, resume_from="prior-checkpoint")
    stage = recipe_module.build_plan(recipe)[0][0]
    assert options(stage)["--resume-from"] == str(ROOT / "prior-checkpoint")
    assert options(stage)["--checkpoint-every"] == "500"
    assert "--init-checkpoint" not in options(stage)
    with pytest.raises(ValueError, match="mutually exclusive"):
        reuse_recipe(tmp_path, resume_from="prior-checkpoint", init_checkpoint="other")


def test_smoke_reused_labels_have_bounded_deterministic_subsets(tmp_path):
    recipe = reuse_recipe(tmp_path, smoke=True)
    stages, _ = recipe_module.build_plan(recipe)
    assert len(stages) == 1
    assert options(stages[0])["--train-data"] == str(tmp_path / "run/smoke_train.jsonl")
    assert options(stages[0])["--steps"] == "2"
    Path(recipe["output_dir"]).mkdir()
    first = recipe_module.write_smoke_inputs(recipe)
    assert first["train_data"]["positions"] == 8
    assert first["eval_data"]["positions"] == 4
    source = Path(recipe["train_data"])
    source.write_text("\n".join(reversed(source.read_text().splitlines())) + "\n")
    assert recipe_module.write_smoke_inputs(recipe) == first


def test_preflight_reuses_teacher_data_without_katago_and_rejects_game_leakage(tmp_path):
    recipe = reuse_recipe(tmp_path)
    resources = recipe_module.preflight(recipe)
    assert set(resources) == {"train_data", "eval_data"}
    rows = [json.loads(line) for line in Path(recipe["eval_data"]).read_text().splitlines()]
    rows[0]["game_id"] = "train-0"
    Path(recipe["eval_data"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="game leakage"):
        recipe_module.preflight(recipe)
    assert not Path(recipe["output_dir"]).exists()


@pytest.mark.parametrize("source,split", [
    ("train_data", "test"), ("train_data", "validation"), ("train_data", None),
    ("eval_data", "test"), ("eval_data", "train"), ("eval_data", None),
])
def test_recipe_requires_pinned_training_roles_without_optional_evaluation(tmp_path, source, split):
    recipe = reuse_recipe(tmp_path)
    assert recipe["evaluation"]["enabled"] is False
    path = Path(recipe[source])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["split"] = split
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=source + " requires explicit"):
        recipe_module.preflight(recipe)
    assert not Path(recipe["output_dir"]).exists()


def test_preflight_rejects_wrong_objective_and_reused_output(tmp_path):
    recipe = reuse_recipe(tmp_path)
    recipe["komi"] = 6.5
    with pytest.raises(ValueError, match="rules/komi"):
        recipe_module.preflight(recipe)
    recipe["komi"] = 7.5
    Path(recipe["output_dir"]).mkdir()
    with pytest.raises(ValueError, match="fresh path"):
        recipe_module.preflight(recipe)


def test_resume_preflight_requires_and_fingerprints_training_state(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    recipe = reuse_recipe(tmp_path, resume_from=str(checkpoint))
    with pytest.raises(ValueError, match="lacks training_state.pt"):
        recipe_module.preflight(recipe)
    (checkpoint / "training_state.pt").write_bytes(b"test-state")
    resources = recipe_module.preflight(recipe)
    assert resources["resume_from/training_state.pt"] == recipe_module.file_hash(checkpoint / "training_state.pt")


@pytest.mark.parametrize("section,field,value", [
    ("training", "batch_size", 0), ("training", "learning_rate", float("nan")),
    ("categorical_training", "readout", "unknown"), ("categorical_training", "value_weight", -1),
    ("text_training", "encoding", "cells"), ("text_training", "generation_positions", 0),
    ("evaluation", "text_num_beams", 0), ("evaluation", "text_num_beams", -1),
    ("evaluation", "text_num_beams", True), ("evaluation", "text_num_beams", 1.5),
])
def test_invalid_training_configuration_fails_before_execution(tmp_path, section, field, value):
    raw = yaml.safe_load((ROOT / "config/policy_19x19_qwen06.yaml").read_text())
    raw[section][field] = value
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        recipe_module.load_recipe(arguments(tmp_path, config=str(path)))


def test_binary_environment_can_name_an_executable_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("KATAGO_BIN", Path(sys.executable).name)
    recipe = recipe_module.load_recipe(arguments(tmp_path))
    assert Path(recipe["katago"]["binary"]).name == Path(sys.executable).name
    assert Path(recipe["katago"]["binary"]).is_file()


@pytest.mark.parametrize("returncode", [0, 7])
def test_stage_logs_and_failure_status_are_preserved(tmp_path, monkeypatch, returncode):
    recipe = reuse_recipe(tmp_path, smoke=True)
    stages, match = recipe_module.build_plan(recipe)

    def execute(argv, cwd, stdout, stderr):
        assert argv == stages[0]["argv"] and cwd == ROOT
        stdout.write("representative stage output\n")
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(recipe_module.subprocess, "run", execute)
    if returncode:
        with pytest.raises(RuntimeError, match="distill failed"):
            recipe_module.run_recipe(recipe, stages, match, str(recipe_module.DEFAULT_CONFIG))
    else:
        recipe_module.run_recipe(recipe, stages, match, str(recipe_module.DEFAULT_CONFIG))
    manifest = json.loads((Path(recipe["output_dir"]) / "manifest.json").read_text())
    assert manifest["status"] == ("failed" if returncode else "complete")
    assert manifest["stages"][0]["returncode"] == returncode
    assert manifest["smoke_inputs"]["train_data"]["positions"] == 8
    assert manifest["input_sha256"]["train_data"]
    assert yaml.safe_load((Path(recipe["output_dir"]) / "recipe.yaml").read_text()) == recipe
    assert manifest["resolved_config_sha256"] == recipe_module.file_hash(Path(recipe["output_dir"]) / "recipe.yaml")
    assert "go_rl/policy_model.py" in manifest["code_sha256"]
    assert (Path(recipe["output_dir"]) / "logs/distill.log").read_text() == "representative stage output\n"
