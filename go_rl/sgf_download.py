"""Download a reproducible starter corpus from KataGo's public bulk archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from sgfmill import sgf

ARCHIVE_NAME = "2025-01-01sgfs.tar.bz2"
ARCHIVE_URL = f"https://katagoarchive.org/kata1/traininggames/{ARCHIVE_NAME}"
ARCHIVE_SHA256 = "98dc06610ccad000ed532fa00c7daf76157d150a1319fb147493c1cf26c5b4ca"
CORPUS_DIRECTORY = "katago_2025_01_01"
DEFAULT_MAX_GAMES = 10000
MAX_SGF_BYTES = 2 * 1024 * 1024


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / ARCHIVE_NAME
    if archive.is_file() and file_sha256(archive) == ARCHIVE_SHA256:
        print(f"Reusing verified SGF archive: {archive}", flush=True)
        return archive
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
        partial = Path(handle.name)
    try:
        print(f"Downloading real training games: {ARCHIVE_URL} (about 21 MiB)", flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                        "--connect-timeout", "30", "--speed-limit", "1024",
                        "--speed-time", "120", "--output", str(partial), ARCHIVE_URL], check=True)
        if file_sha256(partial) != ARCHIVE_SHA256:
            raise RuntimeError("SGF archive checksum mismatch; training data was not installed.")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


def sgf_board_size(data: bytes) -> int:
    """Parse all main-line moves so malformed SGFs aren't installed as data."""
    game = sgf.Sgf_game.from_bytes(data)
    size = game.get_size()
    if size not in (9, 19):
        raise ValueError("Only square 9x9 and 19x19 games are used")
    root = game.get_root()
    if root.has_property("GM") and root.get("GM") != 1:
        raise ValueError("Not a Go game")
    root.get_setup_stones()
    moves = 0
    for node in game.get_main_sequence():
        color, _ = node.get_move()
        if color:
            moves += 1
        if node is not root and node.has_setup_stones():
            raise ValueError("Mid-game setup changes are not supported by the dataset parser")
    if moves < 2:
        raise ValueError("Game contains fewer than two moves")
    return size


def install_games(archive: Path, targets: dict[int, Path], max_games: int) -> dict[int, dict]:
    """Stage validated games, then publish each complete board-size directory.

    Archive paths are never extracted to disk. Files get content-derived names;
    links, non-SGFs, oversized entries, and other board sizes are ignored.
    """
    from contextlib import ExitStack

    with ExitStack() as stack:
        staging = {}
        names: dict[int, set[str]] = {size: set() for size in targets}
        for size, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = stack.enter_context(tempfile.TemporaryDirectory(
                prefix=".sgf-import-", dir=target.parent))
            staging[size] = Path(temporary) / CORPUS_DIRECTORY
            staging[size].mkdir()
        rejected = 0
        with tarfile.open(archive, "r|bz2") as bundle:
            for member in bundle:
                if not member.isfile() or not member.name.lower().endswith(".sgf"):
                    continue
                if member.size > MAX_SGF_BYTES:
                    rejected += 1
                    continue
                with bundle.extractfile(member) as handle:
                    data = handle.read(MAX_SGF_BYTES + 1)
                try:
                    size = sgf_board_size(data)
                except (ValueError, KeyError, UnicodeError):
                    rejected += 1
                    continue
                if size not in targets or len(names[size]) >= max_games:
                    continue
                name = hashlib.sha256(data).hexdigest() + ".sgf"
                (staging[size] / name).write_bytes(data)
                names[size].add(name)
                if all(len(files) >= max_games for files in names.values()):
                    break
        # Validate all requested sizes before publishing any of them.
        for size in targets:
            if not names[size]:
                raise RuntimeError(f"No usable {size}x{size} games in {ARCHIVE_NAME}.")
        reports = {}
        for size, target in targets.items():
            report = {"source_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
                      "board_size": size, "games": len(names[size]),
                      "max_games": max_games, "files": sorted(names[size])}
            (staging[size] / "source.json").write_text(json.dumps(report, indent=2) + "\n")
            target.mkdir(parents=True, exist_ok=True)
            destination = target / CORPUS_DIRECTORY
            if destination.exists():
                raise RuntimeError(f"Incomplete or conflicting corpus at {destination}; move it aside and retry.")
            staging[size].rename(destination)
            reports[size] = {key: value for key, value in report.items() if key != "files"}
            reports[size]["directory"] = str(target.resolve())
            print(f"Installed {len(names[size])} real {size}x{size} games in {destination}", flush=True)
        print(f"Skipped {rejected} malformed, unsupported, or other-board-size SGFs.", flush=True)
        return reports


def ensure_sgf_games(targets: dict[int, Path], cache_dir: Path,
                     max_games: int = DEFAULT_MAX_GAMES) -> dict[int, dict]:
    """Reuse supplied SGFs; download missing board sizes identically on ARM/x86."""
    if max_games < 1 or not targets or any(size not in (9, 19) for size in targets):
        raise ValueError("Request 9x9/19x19 games with a positive max_games limit.")
    if len({path.resolve() for path in targets.values()}) != len(targets):
        raise ValueError("Each board size needs a separate SGF directory.")
    reports = {}
    missing = {}
    for size, path in targets.items():
        if path.exists() and not path.is_dir():
            raise ValueError(f"SGF path is not a directory: {path}")
        files = list(path.rglob("*.sgf")) if path.is_dir() else []
        if files:
            print(f"Reusing {len(files)} existing SGFs in {path}", flush=True)
            reports[size] = {"directory": str(path.resolve()), "games": len(files), "existing": True}
            manifest = path / CORPUS_DIRECTORY / "source.json"
            if manifest.is_file():
                metadata = json.loads(manifest.read_text())
                reports[size]["downloaded_source"] = {
                    key: metadata[key] for key in ("source_url", "archive_sha256", "games")}
        else:
            missing[size] = path
    if missing:
        try:
            reports.update(install_games(fetch_archive(cache_dir), missing, max_games))
        except (OSError, tarfile.TarError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("SGF download/import failed. Retry to reuse the cache, or supply local "
                               "--sgf-9x9 and --sgf-19x19 directories with --no-download-data.") from exc
    return reports
