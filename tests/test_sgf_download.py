from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from go_rl import sgf_download as download


def game(size=9, first="aa"):
    return f"(;GM[1]FF[4]SZ[{size}]KM[7.5];B[{first}];W[bb])".encode()


def archive_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            if content is None:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside.sgf"
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


@pytest.fixture
def source(tmp_path, monkeypatch):
    payload = archive_bytes([
        ("nested/nine.sgf", game(9)), ("duplicate.sgf", game(9)),
        ("../../escape.sgf", game(19)), ("symlink.sgf", None),
        ("other.sgf", game(13)), ("invalid.sgf", b"<html>bad response</html>"),
        ("notes.txt", game(9)), ("second.sgf", game(9, "cc")),
    ])
    monkeypatch.setattr(download, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def fetch(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download.subprocess, "run", fetch)
    return calls


def test_download_installs_both_sizes_safely_and_reuses_existing(tmp_path, source):
    targets = {9: tmp_path / "nine", 19: tmp_path / "nineteen"}
    result = download.ensure_sgf_games(targets, tmp_path / "cache")
    assert len(source) == 1
    assert result[9]["games"] == 2  # Duplicate contents aren't extra games.
    assert result[19]["games"] == 1
    assert not (tmp_path.parent / "escape.sgf").exists()
    for size, target in targets.items():
        for path in target.rglob("*.sgf"):
            assert path.stem == hashlib.sha256(path.read_bytes()).hexdigest()
            assert download.sgf_board_size(path.read_bytes()) == size
        metadata = json.loads((target / download.CORPUS_DIRECTORY / "source.json").read_text())
        assert metadata["archive_sha256"] == download.ARCHIVE_SHA256
    download.ensure_sgf_games(targets, tmp_path / "cache")
    assert len(source) == 1
    assert not list(tmp_path.glob(".sgf-import-*"))


def test_missing_size_reuses_archive_and_preserves_local_games(tmp_path, source):
    local = tmp_path / "local"
    local.mkdir()
    supplied = local / "mine.sgf"
    supplied.write_bytes(game(9))
    targets = {9: local, 19: tmp_path / "nineteen"}
    result = download.ensure_sgf_games(targets, tmp_path / "cache", max_games=1)
    assert result[9]["existing"]
    assert list(local.iterdir()) == [supplied]
    download.ensure_sgf_games({9: tmp_path / "another-nine"}, tmp_path / "cache", max_games=1)
    assert len(source) == 1
    assert len(list((tmp_path / "another-nine").rglob("*.sgf"))) == 1


def test_checksum_mismatch_leaves_no_dataset(tmp_path, source, monkeypatch):
    monkeypatch.setattr(download, "ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="checksum"):
        download.ensure_sgf_games({9: tmp_path / "nine"}, tmp_path / "cache")
    assert not (tmp_path / "nine").exists()
    assert not list((tmp_path / "cache").iterdir())


def test_missing_board_size_publishes_no_partial_corpus(tmp_path):
    archive = tmp_path / "source.tar.bz2"
    archive.write_bytes(archive_bytes([("only-nine.sgf", game(9))]))
    with pytest.raises(RuntimeError, match="19x19"):
        download.install_games(archive, {9: tmp_path / "nine", 19: tmp_path / "nineteen"}, 10)
    assert not (tmp_path / "nine").exists()
    assert not (tmp_path / "nineteen").exists()
    assert not list(tmp_path.glob(".sgf-import-*"))


@pytest.mark.parametrize("contents", [b"not SGF", b"(;SZ[9])", game(13),
    b"(;GM[2]SZ[9];B[aa];W[bb])", b"(;SZ[9];B[aa];AB[bb];W[cc])",
    b"(;SZ[9:19];B[aa];W[bb])", b"(;SZ[9];B[zz];W[bb])"])
def test_rejects_unsupported_sgfs(contents):
    with pytest.raises(ValueError):
        download.sgf_board_size(contents)
