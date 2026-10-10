"""Install a CPU KataGo evaluator without competing for training GPU memory."""
from __future__ import annotations

import os
import platform
import shutil
import tarfile
from pathlib import Path
import shlex
import subprocess
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.15.3"
MODEL = "kata1-b18c384nbt-s7709731328-d3715293823.bin.gz"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted download must not look like a reusable installed file.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        partial = Path(handle.name)
    try:
        print(f"Downloading {url}", flush=True)
        # KataGo's model host can reject urllib's default HTTP client. curl is
        # also used by the environment bootstrap and retries transient failures.
        subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                        "--connect-timeout", "30", "--speed-limit", "1024",
                        "--speed-time", "120", "--output", str(partial), url], check=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def ensure_build_dependencies() -> None:
    """Install the small CPU build toolchain on Debian/Ubuntu when needed."""
    if (shutil.which("cmake") and shutil.which("c++")
            and Path("/usr/include/eigen3/Eigen/Core").is_file()
            and Path("/usr/include/zlib.h").is_file()):
        return
    packages = ["build-essential", "cmake", "libeigen3-dev", "zlib1g-dev"]
    hint = "sudo apt-get update && sudo apt-get install -y " + " ".join(packages)
    if not shutil.which("apt-get"):
        raise SystemExit("Install CMake, a C++ compiler, Eigen3 and zlib development headers, then rerun setup.")
    prefix = []
    if os.geteuid() != 0:
        if not shutil.which("sudo") or subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode:
            raise SystemExit(f"KataGo's native build needs system packages. Run this once, then retry:\n{hint}")
        prefix = ["sudo", "-n"]
    subprocess.run([*prefix, "apt-get", "update"], check=True)
    subprocess.run([*prefix, "apt-get", "install", "-y", *packages], check=True)


def install_binary(binary: Path, machine: str) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    # Build/install in a temporary directory so failed builds never look complete.
    with tempfile.TemporaryDirectory(dir=binary.parent) as temporary:
        work = Path(temporary)
        if machine not in {"x86_64", "aarch64"}:
            raise ValueError(f"Unsupported CPU architecture: {machine}")
        # Older x86 release binaries need libssl.so.1.1. Build against the host's
        # libraries on both platforms instead of depending on that obsolete ABI.
        ensure_build_dependencies()
        archive = work / "source.tar.gz"
        download(f"https://github.com/lightvector/KataGo/archive/refs/tags/v{VERSION}.tar.gz", archive)
        with tarfile.open(archive) as bundle:
            bundle.extractall(work, filter="data")
        source = work / f"KataGo-{VERSION}" / "cpp"
        build = work / "build"
        subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release", "-DUSE_BACKEND=EIGEN",
                        "-DUSE_AVX2=OFF", "-DBUILD_DISTRIBUTED=OFF",
                        "-DNO_GIT_REVISION=ON"], check=True)
        jobs = max(1, min(4, os.cpu_count() or 1))
        subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
        executable = build / "katago"
        executable.chmod(0o755)
        subprocess.run([str(executable), "version"], check=True)
        executable.replace(binary)


def main() -> None:
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in {"x86_64", "aarch64"}:
        raise SystemExit("KataGo setup supports Linux x86_64 and aarch64.")
    install_dir = REPO_ROOT / "katago"
    binary = Path(os.environ.get("KATAGO_BIN", str(install_dir / machine / "katago"))).expanduser().resolve()
    model = Path(os.environ.get("KATAGO_MODEL", str(install_dir / MODEL))).expanduser().resolve()
    for key, path in (("KATAGO_BIN", binary), ("KATAGO_MODEL", model)):
        if key in os.environ and not path.is_file():
            raise SystemExit(f"{key} points to a missing file: {path}. Update .env or unset {key} to install defaults.")
    if not binary.is_file():
        install_binary(binary, machine)
    try:
        subprocess.run([str(binary), "version"], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot execute {binary} on {machine}. Remove the KATAGO_BIN setting from .env to install a native binary.") from exc
    if not model.is_file():
        download(f"https://media.katagotraining.org/uploaded/networks/models/kata1/{MODEL}", model)
    values = {
        "KATAGO_BIN": str(binary),
        "KATAGO_MODEL": str(model),
        "KATAGO_TRAIN_CONFIG": str(REPO_ROOT / "config/katago_analysis_train.cfg"),
        "KATAGO_EVAL_CONFIG": str(REPO_ROOT / "config/katago_analysis_eval.cfg"),
        "KATAGO_CONFIG": str(REPO_ROOT / "config/katago_analysis_eval.cfg"),
    }
    missing = {key: value for key, value in values.items() if key not in os.environ}
    if missing:
        with (REPO_ROOT / ".env").open("a", encoding="utf-8") as handle:
            handle.write("\n# Go RL setup defaults\n")
            for key, value in missing.items():
                handle.write(f"export {key}={shlex.quote(value)}\n")


if __name__ == "__main__":
    main()
