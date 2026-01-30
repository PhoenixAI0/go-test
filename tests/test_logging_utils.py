import logging
from unittest.mock import MagicMock, patch

import pytest

from go_rl.logging_utils import LOG_FORMAT, setup_logging


def test_log_format_defined():
    """Test that LOG_FORMAT is defined."""
    assert isinstance(LOG_FORMAT, str)
    assert "%(asctime)s" in LOG_FORMAT
    assert "%(levelname)s" in LOG_FORMAT
    assert "%(name)s" in LOG_FORMAT
    assert "%(message)s" in LOG_FORMAT


@pytest.mark.parametrize(
    "level_str,expected_level",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("debug", logging.DEBUG),  # lowercase
        ("info", logging.INFO),
        ("warning", logging.WARNING),
    ],
)
def test_setup_logging_sets_level(level_str, expected_level):
    """Test that setup_logging sets the correct log level."""
    with patch("logging.basicConfig") as mock_basic:
        setup_logging(level=level_str)
        mock_basic.assert_called_once()
        assert mock_basic.call_args[1]["level"] == expected_level


def test_setup_logging_default_level():
    """Test that default log level is INFO."""
    with patch("logging.basicConfig") as mock_basic:
        setup_logging()
        mock_basic.assert_called_once()
        assert mock_basic.call_args[1]["level"] == logging.INFO


def test_setup_logging_invalid_level():
    """Test that invalid log level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(level="INVALID")


def test_setup_logging_creates_stream_handler():
    """Test that setup_logging creates a StreamHandler."""
    with patch("logging.basicConfig") as mock_basic:
        setup_logging()
        handlers = mock_basic.call_args[1]["handlers"]
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)


def test_setup_logging_creates_file_handler(tmp_path):
    """Test that setup_logging creates a FileHandler when log_file is provided."""
    log_file = tmp_path / "test.log"

    with patch("logging.basicConfig") as mock_basic:
        setup_logging(log_file=str(log_file))
        handlers = mock_basic.call_args[1]["handlers"]
        assert len(handlers) == 2
        assert isinstance(handlers[0], logging.StreamHandler)
        assert isinstance(handlers[1], logging.FileHandler)


def test_setup_logging_uses_correct_format():
    """Test that setup_logging uses the correct format."""
    with patch("logging.basicConfig") as mock_basic:
        setup_logging()
        assert mock_basic.call_args[1]["format"] == LOG_FORMAT


def test_setup_logging_forces_reconfiguration():
    """Test that setup_logging uses force=True."""
    with patch("logging.basicConfig") as mock_basic:
        setup_logging()
        assert mock_basic.call_args[1]["force"] is True


def test_setup_logging_none_level_defaults_to_info():
    """Test that None level defaults to INFO."""
    with patch("logging.basicConfig") as mock_basic:
        setup_logging(level=None)
        assert mock_basic.call_args[1]["level"] == logging.INFO
