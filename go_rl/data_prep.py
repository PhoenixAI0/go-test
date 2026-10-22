from __future__ import annotations

import logging
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any

from .board import BoardState

logger = logging.getLogger(__name__)


def _find_go_binary() -> str | None:
    go_bin = shutil.which("go")
    if go_bin:
        return go_bin
    local_go = Path.home() / ".local" / "go" / "bin" / "go"
    if local_go.exists():
        return str(local_go)
    return None


def _build_go_parser_binary() -> Path:
    parser_dir = _find_go_parser_dir()
    repo_root = parser_dir.parent
    binary_path = parser_dir / "sgf_parser"

    go_bin = _find_go_binary()
    if not go_bin:
        install_script = repo_root / "scripts" / "install_go.sh"
        if install_script.exists() and os.access(install_script, os.X_OK):
            logger.info("Go toolchain not found, attempting install via %s", install_script)
            try:
                subprocess.run([str(install_script)], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Failed to install Go toolchain required for SGF parsing.") from exc
            go_bin = _find_go_binary()

    if not go_bin:
        raise RuntimeError("Go toolchain is required for dataset parsing but was not found.")

    env = os.environ.copy()
    env.setdefault("GOCACHE", "/tmp/go-build")
    try:
        subprocess.run(
            [go_bin, "build", "-o", str(binary_path), "."],
            cwd=str(parser_dir),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to build Go SGF parser in {parser_dir}") from exc

    if not binary_path.exists() or not os.access(binary_path, os.X_OK):
        raise RuntimeError(f"Go parser binary is missing or not executable: {binary_path}")

    return binary_path


def _find_go_parser_dir() -> Path:
    candidates: list[Path] = []

    file_path = Path(__file__).resolve()
    candidates.extend(parent / "go_parser" for parent in file_path.parents)

    cwd = Path.cwd().resolve()
    candidates.extend(parent / "go_parser" for parent in [cwd, *cwd.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "main.go").exists():
            return candidate

    raise RuntimeError(
        f"Go parser directory not found from module path {file_path} or cwd {cwd}"
    )


def run_go_parser(
    sgf_dir: str,
    output_path: str,
    max_positions: int,
    num_workers: int = 0,
    *,
    seed: int = 42,
    positions_per_game: int = 0,
) -> int:
    if max_positions < 0 or positions_per_game < 0:
        raise ValueError("Position limits must be nonnegative.")
    parser_bin = _build_go_parser_binary()
    workers = num_workers if num_workers > 0 else max(1, (os.cpu_count() or 1) - 1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(parser_bin),
        "-sgf-dir",
        sgf_dir,
        "-output",
        str(output),
        "-max-positions",
        str(max_positions),
        "-workers",
        str(workers),
        "-seed", str(seed),
        "-positions-per-game", str(positions_per_game),
    ]
    logger.info("Running Go SGF parser with %d workers", workers)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Go SGF parser execution failed.") from exc

    with output.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_jsonl_batch(handle, records: Iterable[dict]) -> None:
    batch = list(records)
    if not batch:
        return
    handle.writelines(json.dumps(record) + "\n" for record in batch)


def game_split_indices(
    game_ids: Sequence[str], eval_ratio: float = 0.05, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split whole semantic games, independent of row order and augmentation.

    Hash ordering avoids dependence on Python's salted hash or global RNG. A
    one-game smoke corpus cannot provide an honest held-out game; it has no eval.
    """
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1).")
    if any(not isinstance(game_id, str) or not game_id for game_id in game_ids):
        raise ValueError("Every example needs a nonempty game_id for a game-disjoint split.")
    groups = sorted(set(game_ids), key=lambda value: (
        hashlib.sha256(f"{seed}:{value}".encode()).digest(), value))
    if eval_ratio == 0 or len(groups) < 2:
        return list(range(len(game_ids))), []
    count = min(len(groups) - 1, max(1, round(len(groups) * eval_ratio)))
    held_out = set(groups[:count])
    train, validation = [], []
    for index, game_id in enumerate(game_ids):
        (validation if game_id in held_out else train).append(index)
    return train, validation


def split_dataset_by_game(dataset: Any, eval_ratio: float, seed: int = 42):
    """Select HF dataset splits without letting adjacent game positions leak.

    Prepared datasets may pin a split column. Otherwise game IDs determine it.
    Legacy position-only datasets must be regenerated to enable validation.
    """
    if "split" in dataset.column_names:
        assignments = list(dataset["split"])
        if any(value not in {"train", "validation", "test"} for value in assignments):
            raise ValueError("Unknown dataset split; expected train, validation, or test.")
        if "game_id" not in dataset.column_names:
            raise ValueError("Prepared splits require game_id to audit leakage.")
        seen: dict[str, str] = {}
        for game_id, assignment in zip(dataset["game_id"], assignments):
            if game_id in seen and seen[game_id] != assignment:
                raise ValueError(f"Game {game_id} appears in multiple dataset splits.")
            seen[game_id] = assignment
        train_indices = [i for i, value in enumerate(assignments) if value == "train"]
        eval_indices = [i for i, value in enumerate(assignments) if value == "validation"]
    elif eval_ratio == 0 or len(dataset) < 2:
        return dataset, None
    else:
        if "game_id" not in dataset.column_names:
            raise ValueError("SFT evaluation requires game_id. Regenerate data with prepare_datasets.py "
                             "or explicitly set eval_split_ratio: 0 for a smoke-only run.")
        train_indices, eval_indices = game_split_indices(dataset["game_id"], eval_ratio, seed)
    if not train_indices:
        raise ValueError("Dataset contains no training games.")
    if not eval_indices:
        logger.warning("No held-out games available; validation is disabled.")
    return dataset.select(train_indices), dataset.select(eval_indices) if eval_indices else None


def board_from_example(example: Mapping[str, Any]) -> BoardState:
    """Reconstruct setup plus history, preserving captures and ko for verifiers.

    Legacy records can only reconstruct the visible board. Rich records are
    checked against that board so a corrupted history never silently changes a
    teacher query. Rules and komi remain explicit record fields, not defaults.
    """
    from .board import BoardState, gtp_to_coords
    from .prompting import parse_prompt

    visible = parse_prompt(example["prompt"])
    if "moves" not in example and "initial_stones" not in example:
        return visible
    if "moves" not in example or "initial_stones" not in example:
        raise ValueError("State records require both moves and initial_stones.")
    size = int(example.get("board_size", visible.size))
    if size != visible.size:
        raise ValueError("board_size differs from prompt.")
    matrix = [[0] * size for _ in range(size)]
    for player, move in example["initial_stones"]:
        if player not in {"B", "W"}:
            raise ValueError(f"Unknown setup player: {player}")
        coords = gtp_to_coords(move, size)
        if coords is None or matrix[coords[0]][coords[1]]:
            raise ValueError(f"Invalid or duplicate setup stone: {move}")
        matrix[coords[0]][coords[1]] = 1 if player == "B" else 2
    moves = example["moves"]
    initial_player = (example.get("initial_player") or
                      (moves[0][0] if moves else visible.next_player()))
    board = BoardState.from_matrix(size, matrix, next_player=initial_player)
    for player, move in moves:
        if player != board.next_player():
            raise ValueError("Nonalternating history cannot be replayed as a Go game.")
        board.apply_move(player, move, strict=True)
    if board.current_board() != visible.current_board() or board.next_player() != visible.next_player():
        raise ValueError("History does not reconstruct the prompt state.")
    return board


def state_record_from_board(board: BoardState) -> dict[str, Any]:
    """Serialize board state/history; callers supply rules, komi and provenance.

    Useful when replacing an RL start position: stale SGF history must not be
    left attached to a new prompt. Previous snapshot context is recovered by
    the board's engine serialization path before writing the record.
    """
    from .board import coords_to_gtp

    setup, moves = board.analysis_setup()
    initial_stones = []
    if setup is not None:
        for row, cells in enumerate(setup):
            for col, value in enumerate(cells):
                if value:
                    initial_stones.append(["B" if value == 1 else "W", coords_to_gtp((row, col), board.size)])
    return {
        "prompt": board.to_prompt(), "board_size": board.size,
        "initial_stones": initial_stones,
        "moves": [[player, coords_to_gtp(coords, board.size)] for player, coords in moves],
        "initial_player": moves[0][0] if moves else board.next_player(),
        "to_play": board.next_player(), "move_number": len(moves),
    }


def _transform_coords(coords: tuple[int, int] | None, size: int, symmetry: int) -> tuple[int, int] | None:
    if coords is None:
        return None
    row, col = coords
    if symmetry >= 4:
        col = size - 1 - col
    for _ in range(symmetry % 4):
        row, col = col, size - 1 - row
    return row, col


def _compose_symmetries(first: int, second: int) -> int:
    if first not in range(8) or second not in range(8):
        raise ValueError("symmetry must be in [0, 7].")
    corners = ((0, 0), (0, 1), (1, 0))
    result = [_transform_coords(_transform_coords(point, 2, first), 2, second) for point in corners]
    return next(symmetry for symmetry in range(8)
                if [_transform_coords(point, 2, symmetry) for point in corners] == result)


def transform_example(example: Mapping[str, Any], symmetry: int) -> dict[str, Any]:
    """Apply a D4 symmetry consistently to a grid, action, setup, and history.

    Split before calling; game_id deliberately stays unchanged. Color is never
    swapped because komi makes color exchange an invalid symmetry.
    """
    from .board import BoardState, coords_to_gtp, gtp_to_coords
    from .prompting import parse_prompt

    if symmetry not in range(8):
        raise ValueError("symmetry must be in [0, 7].")
    if any(key in example for key in ("policy_target", "ownership", "legal_mask")):
        raise ValueError("Augment raw game records before adding spatial teacher targets.")
    result = dict(example)
    board = parse_prompt(example["prompt"])
    size = board.size

    def move_label(move: str) -> str:
        return coords_to_gtp(_transform_coords(gtp_to_coords(move, size), size, symmetry), size)

    matrix = [[0] * size for _ in range(size)]
    for row, cells in enumerate(board.current_board()):
        for col, value in enumerate(cells):
            out_row, out_col = _transform_coords((row, col), size, symmetry)
            matrix[out_row][out_col] = value
    result["prompt"] = BoardState.from_matrix(size, matrix, board.next_player()).to_prompt()
    if "response" in result:
        result["response"] = move_label(result["response"])
    if result.get("opponent_response") is not None:
        result["opponent_response"] = move_label(result["opponent_response"])
    for key in ("initial_stones", "moves"):
        if key in result:
            result[key] = [[player, move_label(move)] for player, move in result[key]]
    result["symmetry"] = _compose_symmetries(example.get("symmetry", 0), symmetry)
    return result


def transform_teacher_example(example: Mapping[str, Any], symmetry: int) -> dict[str, Any]:
    """Transform a teacher-labelled record after splitting, without reanalysis.

    Row-major policy/mask entries and all GTP move labels receive the same D4
    mapping; pass stays last. Scalar values, BLACK-perspective engine estimates,
    komi, rules and split/game IDs are invariant. Independent copies prevent an
    augmentation from mutating its source. Supported ownership vectors have one
    entry per intersection and no pass entry.
    """
    from copy import deepcopy
    from .board import coords_to_gtp, gtp_to_coords
    from .prompting import parse_prompt

    if symmetry not in range(8):
        raise ValueError("symmetry must be in [0, 7].")
    spatial_fields = {"policy_target", "legal_mask", "ownership", "ownership_stdev", "ownershipStdev"}
    result = transform_example({key: deepcopy(value) for key, value in example.items()
                                if key not in spatial_fields}, symmetry)
    size = parse_prompt(example["prompt"]).size
    count = size * size
    permutation = []
    for action in range(count):
        row, col = _transform_coords(divmod(action, size), size, symmetry)
        permutation.append(row * size + col)

    def move_label(move: str) -> str:
        return coords_to_gtp(_transform_coords(gtp_to_coords(move, size), size, symmetry), size)

    def vector(values, *, has_pass: bool):
        expected = count + int(has_pass)
        if not isinstance(values, (list, tuple)) or len(values) != expected:
            raise ValueError(f"Spatial teacher vector needs {expected} entries")
        transformed = [None] * expected
        for action, destination in enumerate(permutation):
            transformed[destination] = values[action]
        if has_pass:
            transformed[-1] = values[-1]
        return transformed

    for key in ("policy_target", "legal_mask"):
        if key in example:
            result[key] = vector(example[key], has_pass=True)
    for key in ("ownership", "ownership_stdev", "ownershipStdev"):
        if key in example:
            result[key] = vector(example[key], has_pass=False)
    for key in ("teacher_best_move", "source_response", "source_opponent_response", "learner_move"):
        if result.get(key) is not None:
            result[key] = move_label(result[key])
    for info in result.get("teacher_move_infos", []):
        info["move"] = move_label(info["move"])
        if "isSymmetryOf" in info:
            info["isSymmetryOf"] = move_label(info["isSymmetryOf"])
        if "pv" in info:
            info["pv"] = [move_label(move) for move in info["pv"]]
        for key in ("ownership", "ownershipStdev"):
            if key in info:
                info[key] = vector(info[key], has_pass=False)
    return result


def prepare_dataset_files(
    source: Path, output_dir: Path, tag: str, *, eval_ratio: float = 0.05,
    seed: int = 42, symmetries: int = 1,
) -> dict[str, Any]:
    """Stream metadata-preserving SFT/RL files and pin game-disjoint holdouts.

    Only game IDs are retained in memory. Validation is never augmented and is
    excluded from the RL prompt stream, so both training stages share a holdout.
    """
    if symmetries not in {1, 8}:
        raise ValueError("symmetries must be 1 or 8.")
    game_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                game_ids.add(json.loads(line)["game_id"])
    ordered_ids = sorted(game_ids)
    _, held_out_indices = game_split_indices(ordered_ids, eval_ratio, seed)
    held_out = {ordered_ids[index] for index in held_out_indices}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sft": output_dir / f"sft_{tag}.jsonl",
        "sft_train": output_dir / f"sft_{tag}_train.jsonl",
        "sft_validation": output_dir / f"sft_{tag}_validation.jsonl",
        "rl_train": output_dir / f"rl_prompts_{tag}.jsonl",
        "rl_validation": output_dir / f"rl_prompts_{tag}_validation.jsonl",
    }
    counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    board_counts: Counter[int] = Counter()
    with ExitStack() as stack:
        handles = {key: stack.enter_context(path.open("w", encoding="utf-8", buffering=1024*1024))
                   for key, path in paths.items()}
        with source.open(encoding="utf-8") as source_handle:
            for line in source_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                partition = "validation" if record["game_id"] in held_out else "train"
                record["split"] = partition
                counts[f"source_{partition}"] += 1
                rule_counts[record.get("rules") or "<unspecified>"] += 1
                board_counts[record["board_size"]] += 1
                for symmetry in range(symmetries if partition == "train" else 1):
                    sample = transform_example(record, symmetry) if symmetry else record
                    sample["symmetry"] = symmetry
                    write_jsonl_batch(handles["sft"], [sample])
                    write_jsonl_batch(handles[f"sft_{partition}"], [sample])
                    rl_record = {key: value for key, value in sample.items() if key != "response"}
                    write_jsonl_batch(handles[f"rl_{partition}"], [rl_record])
                    counts[f"written_{partition}"] += 1
    with source.open("rb") as source_handle:
        source_hash = hashlib.file_digest(source_handle, "sha256").hexdigest()
    report = {
        "schema_version": 2, "seed": seed, "eval_ratio": eval_ratio,
        "symmetries": symmetries, "counts": dict(counts),
        "games": {"train": len(game_ids - held_out), "validation": len(held_out)},
        "validation_game_ids": sorted(held_out), "rules": dict(rule_counts),
        "board_sizes": dict(board_counts), "files": {key: str(path) for key, path in paths.items()},
        "source_sha256": source_hash,
    }
    manifest = output_dir / f"dataset_{tag}_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_sft_dataset(
    sgf_dir: str,
    output_path: str,
    max_positions: int,
    num_workers: int = 0,
) -> int:
    logger.info("Building dataset from SGFs with Go parser: %s", sgf_dir)
    count = run_go_parser(
        sgf_dir=sgf_dir,
        output_path=output_path,
        max_positions=max_positions,
        num_workers=num_workers,
    )
    logger.info("Finished writing %d samples to %s", count, output_path)
    return count
