from collections.abc import Sequence
import json
import random

import pytest

from go_rl.experiment_utils import validate_splits
from go_rl.indexed_jsonl import IndexedJsonlRecords, iter_augmented_records, write_record_cache


def test_indexed_rows_preserve_random_sampling_slices_unicode_and_empty_lines(tmp_path):
    rows = [{"index": index, "text": "囲碁", "target": [0.1, 0.2, 0.7]} for index in range(100)]
    path = tmp_path / "records.jsonl"
    path.write_text("\n  \n" + "\n\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    dataset = IndexedJsonlRecords(path)
    assert len(dataset) == len(rows)
    assert dataset.index_bytes == 16 * len(rows)
    assert list(dataset) == rows
    assert dataset[-1] == rows[-1]
    assert dataset[8:2:-2] == rows[8:2:-2]
    rng = random.Random(3407)
    indices = [rng.randrange(len(dataset)) for _ in range(40)]
    assert [dataset[index] for index in indices] == [rows[index] for index in indices]
    selected = dataset.select(iter(indices))
    assert list(selected) == [rows[index] for index in indices]
    assert selected[::-2] == [rows[index] for index in indices][::-2]
    assert selected[-1] == rows[indices[-1]]
    assert selected.index_bytes == 8 * len(indices)
    with pytest.raises(IndexError):
        dataset.select([-1])
    with pytest.raises(IndexError):
        dataset[len(rows)]
    with pytest.raises(IndexError):
        dataset[-len(rows) - 1]


def test_index_refuses_mutation_and_nonobject_records(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"original": true}\n')
    dataset = IndexedJsonlRecords(path)
    path.write_text('{"changed": "larger"}\n')
    with pytest.raises(ValueError, match="changed"):
        dataset[0]
    with pytest.raises(ValueError, match="changed"):
        list(dataset)
    path.write_text("[1, 2, 3]\n")
    with pytest.raises(ValueError, match="must be objects"):
        IndexedJsonlRecords(path)[0]
    path.write_text("\n \n")
    with pytest.raises(ValueError, match="No records"):
        IndexedJsonlRecords(path)


def test_failed_cache_never_publishes_partial_dataset(tmp_path):
    path = tmp_path / "prepared.jsonl"

    def broken_records():
        yield {"tokens": [1, 2, 3]}
        raise ValueError("invalid board")

    with pytest.raises(ValueError, match="invalid board"):
        write_record_cache(broken_records(), path)
    assert not path.exists()
    assert not path.with_suffix(".jsonl.tmp").exists()
    dataset = write_record_cache([{"tokens": [1, 2, 3]}], path)
    assert dataset[0] == {"tokens": [1, 2, 3]}
    with pytest.raises(ValueError, match="already exists"):
        write_record_cache([], path)


def test_d4_is_lazy_and_preserves_original_record_major_sample_order():
    calls = []

    def transform(record, symmetry):
        calls.append((record["record"], symmetry))
        return {**record, "symmetry": symmetry}

    source = [{"record": 0}, {"record": 1}]
    augmented = iter_augmented_records(source, 8, transform)
    assert not calls
    assert next(augmented) == {"record": 0, "symmetry": 0}
    assert calls == [(0, 0)]
    rest = list(augmented)
    assert calls == [(index, symmetry) for index in range(2) for symmetry in range(8)]
    assert rest[-1] == {"record": 1, "symmetry": 7}
    calls.clear()
    assert list(iter_augmented_records(source, 1, transform)) == source
    assert not calls


def test_split_validation_never_materializes_streamed_source_rows(monkeypatch):
    class LiveRow(dict):
        alive = 0
        peak = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            LiveRow.alive += 1
            LiveRow.peak = max(LiveRow.peak, LiveRow.alive)

        def __del__(self):
            LiveRow.alive -= 1

    class StreamingRows(Sequence):
        def __init__(self, partition, count):
            self.partition, self.count = partition, count

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            raise AssertionError("Isolation must stream, not build a random-access row list")

        def __iter__(self):
            for index in range(self.count):
                yield LiveRow(game_id=f"{self.partition}-{index}", split=self.partition)

    monkeypatch.setattr("go_rl.experiment_utils.state_identity", lambda row: row["game_id"])
    validate_splits(StreamingRows("train", 10000), StreamingRows("validation", 20))
    assert LiveRow.peak < 10
