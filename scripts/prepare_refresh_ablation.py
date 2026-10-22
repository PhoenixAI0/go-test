"""Match expert-only and learner-state refresh datasets for offline text imitation.

Both arms retain the same total source-position budget and warm start. Learner
states receive fresh teacher responses; sampled learner actions are not labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path

from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256, position_key, select_game_balanced_indices, state_identity, validate_splits
from go_rl.indexed_jsonl import IndexedJsonlRecords, IndexedRecordSubset, write_record_cache
from go_rl.move_quality import terminal_history
from go_rl.rules import normalize_rules
from go_rl.teacher_data import TARGET_RULES, validate_history_for_rules
from go_rl.training_checkpoint import model_fingerprints


PHASES = ("0", "1-20", "21-60", "61-180", "181-280", "281+")


def validate_holdout_roles(validation: Sequence[dict], test: Sequence[dict]) -> None:
    """The validation file selects checkpoints; the final test is isolation-only."""
    for partition, rows in (("validation", validation), ("test", test)):
        if not rows or any(row.get("split") != partition for row in rows):
            raise ValueError(f"The {partition} source must be nonempty and explicitly pinned to {partition}")
    if {row["game_id"] for row in validation} & {row["game_id"] for row in test}:
        raise ValueError("Validation/final-test game leakage")
    validation_states = {state_identity(row) for row in validation}
    if any(state_identity(row) in validation_states for row in test):
        raise ValueError("Validation/final-test state leakage up to symmetry")


def refresh_phase(row: dict) -> str:
    number = row["move_number"]
    return ("0" if number == 0 else "1-20" if number <= 20 else "21-60" if number <= 60
            else "61-180" if number <= 180 else "181-280" if number <= 280 else "281+")


def phase_matched_replacements(expert: Sequence[dict], shared: Sequence[dict], learner: Sequence[dict],
                               *, seed: int) -> tuple[list[dict], dict]:
    """Match state-coverage quotas using distinct expert rows; report any shortfall."""
    used = {position_key(row) for row in shared}
    candidates = {phase: [] for phase in PHASES}
    requested = Counter(refresh_phase(row) for row in learner)
    for index, row in enumerate(expert):
        if position_key(row) not in used:
            candidates[refresh_phase(row)].append(index)
    replacement, shortages = [], {}
    for phase in PHASES:
        if not requested[phase]:
            continue
        subset = IndexedRecordSubset(expert, candidates[phase])
        chosen = [subset[index] for index in select_game_balanced_indices(subset, requested[phase], seed)]
        replacement.extend(chosen)
        used.update(position_key(row) for row in chosen)
        if len(chosen) < requested[phase]:
            shortages[phase] = requested[phase] - len(chosen)
    fallback_count = len(learner) - len(replacement)
    if fallback_count:
        remainder = IndexedRecordSubset(expert, (index for index, row in enumerate(expert) if position_key(row) not in used))
        replacement.extend(remainder[index] for index in select_game_balanced_indices(remainder, fallback_count, seed))
    if len(replacement) != len(learner):
        raise ValueError("Insufficient distinct expert rows for the phase-matched replacement control")
    actual = Counter(refresh_phase(row) for row in replacement)
    return replacement, {"requested_bins": dict(requested), "selected_bins": dict(actual),
                          "shortages_before_fallback": shortages, "fallback_positions": fallback_count,
                          "exact_phase_match": requested == actual,
                          "repeated_exact_rows": len(replacement) - len({position_key(row) for row in replacement})}


def prepare_arms(expert: Sequence[dict], learner: Sequence[dict], heldouts: Sequence[Sequence[dict]], *,
                 positions: int = 256, learner_positions: int = 64, seed: int = 3407,
                 phase_match_control: bool = False) -> tuple[dict, dict]:
    if positions < 1 or not 1 <= learner_positions < positions:
        raise ValueError("Require a positive learner quota smaller than the total position budget")
    common_teacher = None
    for cohort, rows in (("expert", expert), ("learner", learner)):
        if not rows:
            raise ValueError("Both expert and learner teacher sources must be nonempty")
        for row in rows:
            if row.get("split") != "train" or row.get("teacher_labeled") is not True:
                raise ValueError("Refresh arms require explicitly teacher-labeled training records")
            if cohort == "learner" and row.get("source_type") != "learner_gtp_match":
                raise ValueError("Learner refresh requires provenance from learner GTP match states")
            if row.get("board_size") != 19 or normalize_rules(row["rules"]) != normalize_rules(TARGET_RULES) or row["komi"] != 7.5:
                raise ValueError("Refresh arms require the fixed 19x19 target objective")
            board = board_from_example(row)
            if "moves" not in row or "initial_stones" not in row or row.get("move_number") != len(board.moves):
                raise ValueError("Refresh arms require complete source history")
            if terminal_history(board):
                raise ValueError("Refresh source contains a terminal history")
            validate_history_for_rules(board, row["rules"])
            if row.get("response", "").upper() not in board.legal_moves(superko=True):
                raise ValueError("Teacher response is illegal in its recorded state")
            teacher = row.get("teacher", {})
            contract = {key: teacher[key] for key in (
                "max_visits", "engine_model_sha256", "engine_config_sha256", "normalized_rules", "target_type", "prior_mix")}
            if contract["max_visits"] != 64 or contract["normalized_rules"] != normalize_rules(row["rules"]):
                raise ValueError("Refresh labels require the fixed 64-visit teacher and objective")
            if common_teacher is not None and contract != common_teacher:
                raise ValueError("Expert and learner sources must share the same teacher budget/network/config/objective")
            common_teacher = contract
        for heldout in heldouts:
            validate_splits(rows, heldout)
    selected_expert = [expert[index] for index in select_game_balanced_indices(expert, positions, seed)]
    selected_learner = [learner[index] for index in select_game_balanced_indices(learner, learner_positions, seed)]
    if len(selected_expert) != positions:
        raise ValueError("Insufficient distinct expert positions for the matched control")
    shared_expert = selected_expert[:positions - len(selected_learner)]
    phase_match = None
    if phase_match_control:
        replacement, phase_match = phase_matched_replacements(expert, shared_expert, selected_learner, seed=seed)
        selected_expert = [*shared_expert, *replacement]
    arms = {"expert_control": selected_expert, "learner_mix": [*shared_expert, *selected_learner]}
    summary = {"positions_per_arm": positions, "learner_requested": learner_positions,
               "learner_selected": len(selected_learner), "shared_expert_positions": len(shared_expert),
               "teacher_contract": common_teacher, "seed": seed, "phase_matched_control": phase_match, "arms": {}}
    for name, rows in arms.items():
        games = Counter(row["game_id"] for row in rows)
        identities = {state_identity(row) for row in rows}
        summary["arms"][name] = {
            "positions": len(rows), "games": len(games), "positions_by_game": dict(sorted(games.items())),
            "effective_games_by_position_weight": len(rows) ** 2 / sum(count * count for count in games.values()),
            "unique_visible_states": len(identities), "repeated_visible_states": len(rows) - len(identities),
            "teacher_pass_targets": sum(row["response"].upper() == "PASS" for row in rows),
            "move_bins": dict(Counter(refresh_phase(row) for row in rows)),
            "selected_positions": [{"game_id": row["game_id"], "move_number": row["move_number"]} for row in rows],
        }
    summary["learner_game_counts"] = dict(Counter(row["game_id"] for row in selected_learner))
    return arms, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-data", required=True)
    parser.add_argument("--learner-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--warm-start-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-match-control", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("Output directory exists; choose a fresh experiment path")
    sources = {name: IndexedJsonlRecords(path) for name, path in (
        ("expert", args.expert_data), ("learner", args.learner_data),
        ("validation", args.validation_data), ("test", args.test_data))}
    validate_holdout_roles(sources["validation"], sources["test"])
    arms, manifest = prepare_arms(sources["expert"], sources["learner"], [sources["validation"], sources["test"]],
                                  seed=args.seed, phase_match_control=args.phase_match_control)
    checkpoint = Path(args.warm_start_checkpoint)
    if json.loads((checkpoint / "go_text.json").read_text()).get("encoding") != "compact":
        raise ValueError("This controlled recipe requires the same compact text warm start")
    manifest.update(purpose="controlled_offline_learner_state_refresh_imitation", arguments=vars(args),
                    source_sha256={name: file_sha256(rows.path) for name, rows in sources.items()},
                    warm_start_files_sha256=model_fingerprints(checkpoint),
                    selection="same deterministic expert prefix in both arms; optional phase-matched expert replacements for learner rows",
                    selection_uses_teacher_quality=False, new_engine_or_model_queries=False,
                    limitation="Offline teacher-response cross-entropy on learner-visited states; not on-policy RL or a fresh independent holdout")
    output.mkdir(parents=True)
    commands = {}
    for name, rows in arms.items():
        path = output / f"{name}.jsonl"
        write_record_cache(rows, path)
        manifest["arms"][name]["dataset_sha256"] = file_sha256(path)
        commands[name] = ["uv", "run", "--locked", "python", "scripts/run_text_distill.py",
                          "--train-data", str(path), "--eval-data", args.validation_data,
                          "--init-checkpoint", str(checkpoint), "--encoding", "compact", "--symmetries", "8",
                          "--steps", "120", "--batch-size", "4", "--gradient-accumulation", "1",
                          "--learning-rate", "1e-4", "--seed", str(args.seed), "--eval-every", "20",
                          "--output-dir", f"outputs/research/refresh_{name}_s{args.seed}"]
    manifest["training_commands"] = commands
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "training_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(json.dumps({"output": str(output), "positions_per_arm": manifest["positions_per_arm"],
                      "learner_selected": manifest["learner_selected"], "learner_game_counts": manifest["learner_game_counts"],
                      "dataset_hashes": {name: data["dataset_sha256"] for name, data in manifest["arms"].items()}}))


if __name__ == "__main__":
    main()
