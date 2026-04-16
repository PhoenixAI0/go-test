import json
from unittest.mock import MagicMock, patch

from go_rl.run_logging import (
    _gpu_info,
    _package_version,
    collect_env_info,
    write_run_log,
)


def test_package_version_found():
    """Test that _package_version returns version for installed packages."""
    with patch("go_rl.run_logging.metadata.version", return_value="1.2.3"):
        assert _package_version("some-package") == "1.2.3"


def test_package_version_not_found():
    """Test that _package_version returns 'missing' for missing packages."""
    from importlib import metadata

    with patch(
        "go_rl.run_logging.metadata.version",
        side_effect=metadata.PackageNotFoundError,
    ):
        assert _package_version("nonexistent-package") == "missing"


def test_gpu_info_no_torch():
    """Test _gpu_info when torch is not installed."""
    with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
        info = _gpu_info()
        assert info["available"] is False
        assert "torch not installed" in info["detail"]


def test_gpu_info_cuda_available():
    """Test _gpu_info when CUDA is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2

    mock_props = MagicMock()
    mock_props.name = "NVIDIA GeForce RTX 3080"
    mock_props.total_memory = 10737418240  # 10GB

    mock_torch.cuda.get_device_properties.return_value = mock_props
    mock_torch.version.cuda = "11.8"

    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = _gpu_info()

    assert info["available"] is True
    assert info["backend"] == "cuda"
    assert info["device_count"] == 2
    assert info["name"] == "NVIDIA GeForce RTX 3080"
    assert info["total_memory_bytes"] == 10737418240
    assert info["torch_cuda"] == "11.8"


def test_gpu_info_mps_available():
    """Test _gpu_info when Apple MPS is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_built.return_value = True
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.version.cuda = None

    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = _gpu_info()

    assert info["available"] is True
    assert info["backend"] == "mps"
    assert info["name"] == "Apple MPS"
    assert info["torch_mps_built"] is True
    assert info["torch_mps_available"] is True


def test_gpu_info_no_cuda_no_mps():
    """Test _gpu_info when neither CUDA nor MPS is available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_built.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.version.cuda = None

    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = _gpu_info()

    assert info["available"] is False
    assert "cuda/mps not available" in info["detail"]
    assert info["torch_mps_built"] is False
    assert info["torch_mps_available"] is False


def test_gpu_info_mps_exception():
    """Test _gpu_info when MPS check raises an exception."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_built.side_effect = RuntimeError("MPS error")
    mock_torch.version.cuda = None

    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = _gpu_info()

    assert info["available"] is False
    assert info["torch_mps_built"] is False
    assert info["torch_mps_available"] is False


def test_collect_env_info():
    """Test collect_env_info returns expected structure."""
    packages = ["numpy", "torch"]

    with patch("go_rl.run_logging._package_version") as mock_version, patch(
        "go_rl.run_logging._gpu_info"
    ) as mock_gpu, patch("sys.version", "3.12.0 (main)"), patch(
        "platform.platform", return_value="Linux-5.15.0"
    ), patch(
        "os.getcwd", return_value="/home/user/project"
    ):
        mock_version.side_effect = lambda name: f"{name}-1.0.0"
        mock_gpu.return_value = {"available": True, "backend": "cuda"}

        info = collect_env_info(packages)

    assert "python" in info
    assert info["python"] == "3.12.0"
    assert info["platform"] == "Linux-5.15.0"
    assert info["cwd"] == "/home/user/project"
    assert "packages" in info
    assert info["packages"]["numpy"] == "numpy-1.0.0"
    assert info["packages"]["torch"] == "torch-1.0.0"
    assert info["gpu"]["available"] is True
    assert info["gpu"]["backend"] == "cuda"


def test_collect_env_info_empty_packages():
    """Test collect_env_info with no packages."""
    with patch("go_rl.run_logging._gpu_info") as mock_gpu, patch(
        "sys.version", "3.12.0"
    ), patch("platform.platform", return_value="Linux"), patch(
        "os.getcwd", return_value="/home"
    ):
        mock_gpu.return_value = {"available": False}

        info = collect_env_info([])

    assert info["packages"] == {}
    assert "gpu" in info


def test_write_run_log(tmp_path):
    """Test write_run_log creates a JSON file with correct content."""
    log_dir = tmp_path / "logs"
    run_name = "test_run"
    config_path = "/config/test.yaml"
    config_snapshot = {"learning_rate": 0.001, "batch_size": 32}

    with patch("go_rl.run_logging.collect_env_info") as mock_env, patch(
        "go_rl.run_logging.datetime"
    ) as mock_datetime:
        mock_env.return_value = {
            "python": "3.12.0",
            "packages": {},
            "gpu": {"available": False},
        }
        mock_datetime.utcnow.return_value.strftime.return_value = "20260130_120000"

        log_path = write_run_log(
            str(log_dir),
            run_name,
            config_path,
            config_snapshot,
        )

    # Verify file was created
    assert log_dir.exists()
    assert log_path == str(log_dir / "test_run_20260130_120000.json")

    # Verify content
    with open(log_path, "r") as f:
        content = json.load(f)

    assert content["run_name"] == run_name
    assert content["timestamp_utc"] == "20260130_120000"
    assert content["config_path"] == config_path
    assert content["config"] == config_snapshot
    assert "environment" in content
    assert content["environment"]["python"] == "3.12.0"


def test_write_run_log_with_extra(tmp_path):
    """Test write_run_log includes extra fields."""
    log_dir = tmp_path / "logs"
    extra = {"experiment_id": "exp_001", "notes": "test"}

    with patch("go_rl.run_logging.collect_env_info") as mock_env, patch(
        "go_rl.run_logging.datetime"
    ) as mock_datetime:
        mock_env.return_value = {"python": "3.12.0"}
        mock_datetime.utcnow.return_value.strftime.return_value = "20260130_120000"

        log_path = write_run_log(
            str(log_dir),
            "test_run",
            "/config/test.yaml",
            {},
            extra=extra,
        )

    with open(log_path, "r") as f:
        content = json.load(f)

    assert "extra" in content
    assert content["extra"]["experiment_id"] == "exp_001"
    assert content["extra"]["notes"] == "test"


def test_write_run_log_with_packages(tmp_path):
    """Test write_run_log collects package versions."""
    log_dir = tmp_path / "logs"
    packages = ["numpy", "torch"]

    with patch("go_rl.run_logging.collect_env_info") as mock_env, patch(
        "go_rl.run_logging.datetime"
    ) as mock_datetime:
        mock_env.return_value = {
            "python": "3.12.0",
            "packages": {"numpy": "1.24.0", "torch": "2.0.0"},
        }
        mock_datetime.utcnow.return_value.strftime.return_value = "20260130_120000"

        log_path = write_run_log(
            str(log_dir),
            "test_run",
            "/config/test.yaml",
            {},
            packages=packages,
        )

    # Verify collect_env_info was called with packages
    mock_env.assert_called_once_with(packages)

    with open(log_path, "r") as f:
        content = json.load(f)

    assert content["environment"]["packages"]["numpy"] == "1.24.0"
    assert content["environment"]["packages"]["torch"] == "2.0.0"


def test_write_run_log_creates_directory(tmp_path):
    """Test that write_run_log creates the log directory if it doesn't exist."""
    log_dir = tmp_path / "nested" / "logs" / "dir"

    with patch("go_rl.run_logging.collect_env_info") as mock_env, patch(
        "go_rl.run_logging.datetime"
    ) as mock_datetime:
        mock_env.return_value = {"python": "3.12.0"}
        mock_datetime.utcnow.return_value.strftime.return_value = "20260130_120000"

        write_run_log(str(log_dir), "test_run", "/config/test.yaml", {})

    assert log_dir.exists()
    assert (log_dir / "test_run_20260130_120000.json").exists()


def test_write_run_log_json_formatting(tmp_path):
    """Test that JSON is formatted with indentation and sorted keys."""
    log_dir = tmp_path / "logs"

    with patch("go_rl.run_logging.collect_env_info") as mock_env, patch(
        "go_rl.run_logging.datetime"
    ) as mock_datetime:
        mock_env.return_value = {"python": "3.12.0"}
        mock_datetime.utcnow.return_value.strftime.return_value = "20260130_120000"

        log_path = write_run_log(
            str(log_dir), "test_run", "/config/test.yaml", {"z": 1, "a": 2}
        )

    # Read raw content to check formatting
    with open(log_path, "r") as f:
        raw_content = f.read()

    # Should have indentation
    assert "  " in raw_content  # 2-space indent
    # Should have sorted keys (config.a before config.z)
    assert raw_content.index('"a"') < raw_content.index('"z"')
