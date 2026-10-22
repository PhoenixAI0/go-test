from collections import defaultdict
from collections.abc import Sequence
import json
import random
import tracemalloc

import pytest

from go_rl.experiment_utils import stable_digest
from go_rl.indexed_jsonl import IndexedJsonlRecords
from go_rl.teacher_data import select_game_balanced_indices, select_phase_balanced_indices, move_phase


def old_phase_selection(records, count, seed, per_game):
    """Original full-materialization specification for the memory-only rewrite."""
    phases = ["0-60", "61-180", "181-280", "281+"]
    groups = defaultdict(lambda: defaultdict(dict))
    for index, row in enumerate(records):
        key = stable_digest([row["game_id"], row["move_number"], row.get("prompt"), row.get("moves"),
                             row.get("initial_stones"), row.get("rules"), row.get("komi")])
        groups[row["game_id"]][move_phase(row["move_number"])][key] = index
    games = sorted(groups, key=lambda game: (stable_digest([seed, "game", game]), game))
    ordered = {}
    for game in games:
        queues = {phase: [bucket[key] for key in sorted(bucket, key=lambda key: (
            stable_digest([seed, "position", key]), key))] for phase, bucket in groups[game].items()}
        offset = int(stable_digest([seed, "phase-offset", game])[:16], 16) % 4
        cycle = phases[offset:] + phases[:offset]
        selected = []
        while any(queues.values()) and len(selected) < per_game:
            for phase in cycle:
                if queues.get(phase) and len(selected) < per_game:
                    selected.append(queues[phase].pop(0))
        ordered[game] = selected
    return [ordered[game][index] for index in range(per_game) for game in games
            if index < len(ordered[game])][:count]


@pytest.mark.parametrize("count,per_game", [(1, 1), (1, 30), (7, 5), (25, 5), (1000, 30)])
def test_bounded_reservoir_matches_original_including_duplicates_and_phase_exhaustion(count, per_game):
    rows = [{"game_id": str(game), "move_number": move, "prompt": f"{game}-{move}", "response": "first"}
            for game in range(7) for move in range(10, 400, 7 + game)]
    rows += [{**row, "response": "last"} for row in rows[::9]]
    for seed in (1, 42, 3407):
        assert select_phase_balanced_indices(rows, count, seed, per_game) == old_phase_selection(
            rows, count, seed, per_game)


def test_indexed_game_shuffle_matches_original_and_filters_without_copying_rows(tmp_path):
    rows = [{"game_id": str(game), "move_number": move, "split": "train" if move % 2 else "validation"}
            for game in range(7) for move in range(10)]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = IndexedJsonlRecords(path)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            groups[row["game_id"]].append(index)
    rng = random.Random(3407)
    games = sorted(groups)
    rng.shuffle(games)
    for game in games:
        rng.shuffle(groups[game])
    expected = [groups[game][index] for index in range(3) for game in games][:15]
    actual = select_game_balanced_indices(source, 15, 3407, 3, predicate=lambda row: row["split"] == "train")
    assert actual == expected
    assert list(source.select(actual)) == [rows[index] for index in expected]


def test_phase_reservoir_does_not_retain_large_decoded_source_rows():
    class LargeRows(Sequence):
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError
            return {"game_id": "one", "move_number": index, "prompt": str(index) + "." * 32768}

    tracemalloc.start()
    try:
        indices = select_phase_balanced_indices(LargeRows(), 8, 3407, 8)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(indices) == 8
    assert peak < 2_000_000  # Materializing these source rows requires over 32 MB.
