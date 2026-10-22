"""Read immutable JSONL datasets with compact offsets instead of resident rows.

Prepared caches contain tokens/targets only. Large source histories and their
D4 copies can be transformed one at a time without changing sampling order.
"""
from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import json
import operator
from pathlib import Path
from typing import Any


class IndexedJsonlRecords(Sequence[dict[str, Any]]):
    """Random-access JSONL with 16 bytes of offsets per nonempty record.

    Iteration streams one open file. Random lookup opens/seeks/reads one row;
    there are no persistent file descriptors, parsed-row caches, or whole-file
    reads. Inputs must not be changed while a run is using the index.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offsets = array("Q")
        self._lengths = array("Q")
        self._signature = self._file_signature()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)
                    self._lengths.append(len(line))
        self._check_unchanged()
        if not self._offsets:
            raise ValueError(f"No records in {self.path}")

    def _file_signature(self) -> tuple[int, int, int, int]:
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _check_unchanged(self) -> None:
        if self._file_signature() != self._signature:
            raise ValueError(f"Indexed dataset changed during the run: {self.path}")

    def _decode(self, line: bytes) -> dict[str, Any]:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL records must be objects: {self.path}")
        return row

    @property
    def index_bytes(self) -> int:
        return len(self._offsets) * (self._offsets.itemsize + self._lengths.itemsize)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("JSONL record index out of range")
        self._check_unchanged()
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.read(self._lengths[index])
        return self._decode(line)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._check_unchanged()
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    yield self._decode(line)
        self._check_unchanged()

    def select(self, indices: Iterable[int]) -> IndexedRecordSubset:
        return IndexedRecordSubset(self, indices)


class IndexedRecordSubset(Sequence[dict[str, Any]]):
    """Preserve a selected row order using compact indices, not decoded rows."""

    def __init__(self, source: Sequence[dict[str, Any]], indices: Iterable[int]):
        self.source = source
        self._indices = array("Q")
        for raw in indices:
            index = operator.index(raw)
            if not 0 <= index < len(source):
                raise IndexError("Subset index out of range")
            self._indices.append(index)

    @property
    def index_bytes(self) -> int:
        return len(self._indices) * self._indices.itemsize

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self.source[position] for position in self._indices[index]]
        return self.source[self._indices[index]]


def iter_augmented_records(
    records: Iterable[Mapping[str, Any]], symmetries: int,
    transform: Callable[[Mapping[str, Any], int], dict[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    """Match the original record-major D4 list order with one live copy."""
    if symmetries not in (1, 8):
        raise ValueError("symmetries must be 1 or 8")
    for record in records:
        if symmetries == 1:
            yield record
        else:
            for symmetry in range(8):
                yield transform(record, symmetry)


def write_record_cache(records: Iterable[Mapping[str, Any]], path: str | Path) -> IndexedJsonlRecords:
    """Atomically write prepared records, without retaining source/decoded rows."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise ValueError(f"Prepared cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    created_temporary = False
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            created_temporary = True
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
        if not count:
            raise ValueError("Prepared cache contains no records")
        temporary.replace(path)
    except BaseException:
        if created_temporary:
            temporary.unlink(missing_ok=True)
        raise
    return IndexedJsonlRecords(path)
