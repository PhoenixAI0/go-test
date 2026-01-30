import os
from unittest.mock import MagicMock, patch

import pytest

from go_rl.training_utils import (
    configure_torch_for_speed,
    default_num_workers,
    quantization_kwargs,
)


@pytest.mark.parametrize(
    "bits,expected",
    [
        (4, {"load_in_4bit": True}),
        (8, {"load_in_8bit": True}),
        (0, {}),
        (16, {}),
        (32, {}),
    ],
)
def test_quantization_kwargs_valid_bits(bits, expected):
    """Test that valid quantization bits return correct kwargs."""
    assert quantization_kwargs(bits) == expected


def test_quantization_kwargs_invalid_bits():
    """Test that invalid quantization bits raise ValueError."""
    with pytest.raises(ValueError, match="quantization_bits must be one of"):
        quantization_kwargs(3)

    with pytest.raises(ValueError, match="quantization_bits must be one of"):
        quantization_kwargs(7)

    with pytest.raises(ValueError, match="quantization_bits must be one of"):
        quantization_kwargs(12)


def test_default_num_workers_basic():
    """Test worker count calculation with various CPU counts."""
    with patch("os.cpu_count", return_value=16):
        assert default_num_workers() == 8  # 16 // 2

    with patch("os.cpu_count", return_value=8):
        assert default_num_workers() == 4  # 8 // 2

    with patch("os.cpu_count", return_value=4):
        assert default_num_workers() == 2  # 4 // 2

    with patch("os.cpu_count", return_value=2):
        assert default_num_workers() == 1  # 2 // 2


def test_default_num_workers_caps_at_8():
    """Test that worker count is capped at 8."""
    with patch("os.cpu_count", return_value=32):
        assert default_num_workers() == 8  # max(1, min(8, 16)) = 8

    with patch("os.cpu_count", return_value=64):
        assert default_num_workers() == 8  # max(1, min(8, 32)) = 8


def test_default_num_workers_minimum_1():
    """Test that worker count has a minimum of 1."""
    with patch("os.cpu_count", return_value=1):
        assert default_num_workers() == 1  # max(1, min(8, 0 or 1)) = 1

    with patch("os.cpu_count", return_value=None):
        assert default_num_workers() == 1  # Handle None from os.cpu_count()


def test_configure_torch_for_speed_no_torch():
    """Test that function handles missing torch gracefully."""
    with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
        # Should not raise
        configure_torch_for_speed()


def test_configure_torch_for_speed_no_cuda():
    """Test that function handles no CUDA gracefully."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Should not raise when CUDA is unavailable
        configure_torch_for_speed()

    # Function should return early without setting optimizations


def test_configure_torch_for_speed_with_cuda():
    """Test that function enables CUDA optimizations when available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.cuda.matmul.allow_tf32 = False
    mock_torch.backends.cudnn.allow_tf32 = False
    mock_torch.backends.cudnn.benchmark = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        configure_torch_for_speed()

    # Verify TF32 was enabled
    assert mock_torch.backends.cuda.matmul.allow_tf32 is True
    assert mock_torch.backends.cudnn.allow_tf32 is True
    assert mock_torch.backends.cudnn.benchmark is True


def test_configure_torch_for_speed_with_matmul_precision():
    """Test that matmul precision is set when available."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.set_float32_matmul_precision = MagicMock()

    with patch.dict("sys.modules", {"torch": mock_torch}):
        configure_torch_for_speed()

    mock_torch.set_float32_matmul_precision.assert_called_once_with("high")


def test_configure_torch_for_speed_without_cudnn():
    """Test that function handles missing cudnn gracefully."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    # Remove cudnn attribute
    del mock_torch.backends.cudnn

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Should not raise even without cudnn
        configure_torch_for_speed()

    # Should still set matmul TF32
    assert mock_torch.backends.cuda.matmul.allow_tf32 is True


def test_configure_torch_for_speed_without_matmul_precision():
    """Test that function handles missing set_float32_matmul_precision gracefully."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    # Remove set_float32_matmul_precision attribute
    del mock_torch.set_float32_matmul_precision

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Should not raise even without this method
        configure_torch_for_speed()

    # Should still set other optimizations
    assert mock_torch.backends.cuda.matmul.allow_tf32 is True
