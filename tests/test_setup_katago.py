"""Check architecture selection and that interrupted installs are not reused."""
import importlib.util
from pathlib import Path
import subprocess
import tarfile

import pytest


@pytest.fixture
def installer():
    path = Path(__file__).resolve().parents[1] / "scripts/setup_katago.py"
    spec = importlib.util.spec_from_file_location("setup_katago", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_install_native_binary(installer, tmp_path, monkeypatch, machine):
    commands = []
    urls = []
    monkeypatch.setattr(installer, "ensure_build_dependencies", lambda: None)

    def download(url, destination):
        urls.append(url)
        source = tmp_path / f"KataGo-{installer.VERSION}" / "cpp"
        source.mkdir(parents=True)
        (source / "CMakeLists.txt").write_text("# source fixture")
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(source.parent, arcname=source.parent.name)

    def run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["cmake", "--build"]:
            build = Path(command[2])
            build.mkdir()
            (build / "katago").write_bytes(b"native executable")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer, "download", download)
    monkeypatch.setattr(installer.subprocess, "run", run)
    binary = tmp_path / machine / "katago"
    installer.install_binary(binary, machine)
    assert binary.read_bytes() == b"native executable"
    assert binary.stat().st_mode & 0o111
    assert len(urls) == 1
    assert urls[0].endswith(".tar.gz")
    assert "-DUSE_BACKEND=EIGEN" in commands[0]
    assert "-DUSE_AVX2=OFF" in commands[0]
    assert "-DNO_GIT_REVISION=ON" in commands[0]
    assert commands[-1][1] == "version"


def test_failed_install_does_not_leave_executable(installer, tmp_path, monkeypatch):
    def fail():
        raise subprocess.CalledProcessError(1, "build dependencies")

    monkeypatch.setattr(installer, "ensure_build_dependencies", fail)
    binary = tmp_path / "x86_64" / "katago"
    with pytest.raises(subprocess.CalledProcessError):
        installer.install_binary(binary, "x86_64")
    assert not binary.exists()
    assert list(binary.parent.iterdir()) == []


def test_failed_download_keeps_existing_file(installer, tmp_path, monkeypatch):
    destination = tmp_path / "model.bin.gz"
    destination.write_bytes(b"previous valid model")

    def fail(command, **kwargs):
        partial = Path(command[command.index("--output") + 1])
        partial.write_bytes(b"interrupted download")
        raise subprocess.CalledProcessError(18, command)

    monkeypatch.setattr(installer.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        installer.download("https://example.invalid/model", destination)
    assert destination.read_bytes() == b"previous valid model"
    assert list(tmp_path.iterdir()) == [destination]
