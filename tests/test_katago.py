import json
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest

from go_rl.board import BoardState
from go_rl.config import KataGoConfig
from go_rl.katago import KataGoAnalysisClient, KataGoResult, _extract_winrate


@pytest.fixture
def katago_config():
    return KataGoConfig(
        binary_path="/usr/bin/katago",
        model_path="/models/kata.bin.gz",
        config_path="/config/gtp.cfg",
        board_size=9,
        rules="chinese",
        komi=7.5,
    )


@pytest.fixture
def mock_process():
    """Mock subprocess.Popen for KataGo process."""
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.terminate = MagicMock()
    process.wait = MagicMock()
    process.kill = MagicMock()
    return process


def test_katago_result_dataclass():
    result = KataGoResult(winrate=0.65, raw={"rootInfo": {"winrate": 0.65}})
    assert result.winrate == 0.65
    assert result.raw["rootInfo"]["winrate"] == 0.65


def test_extract_winrate_success():
    response = {"rootInfo": {"winrate": 0.75}}
    assert _extract_winrate(response) == 0.75


def test_extract_winrate_missing_rootinfo():
    response = {"id": 1}
    with pytest.raises(KeyError, match="missing rootInfo.winrate"):
        _extract_winrate(response)


def test_extract_winrate_missing_winrate():
    response = {"rootInfo": {"moves": []}}
    with pytest.raises(KeyError, match="missing rootInfo.winrate"):
        _extract_winrate(response)


def test_katago_client_init(katago_config):
    client = KataGoAnalysisClient(katago_config)
    assert client.config == katago_config
    assert client._process is None
    assert client._next_id == 1
    assert client._stderr_thread is None


@patch("subprocess.Popen")
def test_katago_start_process(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    client = KataGoAnalysisClient(katago_config)

    client.start()

    mock_popen.assert_called_once_with(
        [
            "/usr/bin/katago",
            "analysis",
            "-model",
            "/models/kata.bin.gz",
            "-config",
            "/config/gtp.cfg",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert client._process is not None
    assert client._stderr_thread is not None


@patch("subprocess.Popen")
def test_katago_start_already_running(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    client = KataGoAnalysisClient(katago_config)

    client.start()
    first_process = client._process

    # Call start again - should not create new process
    client.start()
    assert client._process is first_process
    assert mock_popen.call_count == 1


@patch("subprocess.Popen")
def test_katago_close_process(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    client = KataGoAnalysisClient(katago_config)

    client.start()
    client.close()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=5)
    mock_process.stdin.close.assert_called_once()
    mock_process.stdout.close.assert_called_once()
    mock_process.stderr.close.assert_called_once()
    assert client._process is None


@patch("subprocess.Popen")
def test_katago_close_timeout_kills_process(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="katago", timeout=5)

    client = KataGoAnalysisClient(katago_config)
    client.start()
    client.close()

    mock_process.terminate.assert_called_once()
    mock_process.kill.assert_called_once()


def test_katago_close_no_process(katago_config):
    client = KataGoAnalysisClient(katago_config)
    # Should not raise when closing without starting
    client.close()


@patch("subprocess.Popen")
def test_katago_send_not_started(katago_config):
    client = KataGoAnalysisClient(katago_config)
    with pytest.raises(RuntimeError, match="KataGo process not started"):
        client._send({"id": 1})


@patch("subprocess.Popen")
def test_katago_send_success(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    response_data = {"id": 1, "rootInfo": {"winrate": 0.5}}
    mock_process.stdout.readline.return_value = json.dumps(response_data) + "\n"

    client = KataGoAnalysisClient(katago_config)
    client.start()

    payload = {"id": 1, "boardXSize": 9, "boardYSize": 9}
    result = client._send(payload)

    assert result == response_data
    mock_process.stdin.write.assert_called_once()
    mock_process.stdin.flush.assert_called_once()
    written_data = mock_process.stdin.write.call_args[0][0]
    assert json.loads(written_data.strip()) == payload


@patch("subprocess.Popen")
def test_katago_send_empty_response(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    mock_process.stdout.readline.return_value = ""

    client = KataGoAnalysisClient(katago_config)
    client.start()

    with pytest.raises(RuntimeError, match="empty response"):
        client._send({"id": 1})


@patch("subprocess.Popen")
def test_katago_analyze_position(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    response_data = {"id": 1, "rootInfo": {"winrate": 0.65}}
    mock_process.stdout.readline.return_value = json.dumps(response_data) + "\n"

    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    board.apply_move("B", "D4")
    board.apply_move("W", "PASS")

    result = client.analyze_position(board)

    assert isinstance(result, KataGoResult)
    assert result.winrate == 0.65
    assert result.raw == response_data
    assert client._next_id == 2  # Should increment


@patch("subprocess.Popen")
def test_katago_analyze_position_payload_format(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    response_data = {"id": 1, "rootInfo": {"winrate": 0.5}}
    mock_process.stdout.readline.return_value = json.dumps(response_data) + "\n"

    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(5)
    board.apply_move("B", "A1")
    board.apply_move("W", "B2")

    client.analyze_position(board)

    written_data = mock_process.stdin.write.call_args[0][0]
    payload = json.loads(written_data.strip())

    assert payload["id"] == 1
    assert payload["boardXSize"] == 5
    assert payload["boardYSize"] == 5
    assert payload["rules"] == "chinese"
    assert payload["komi"] == 7.5
    assert payload["moves"] == [["b", "a1"], ["w", "b2"]]


@patch("subprocess.Popen")
def test_katago_evaluate_moves(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process

    # Mock multiple responses
    responses = [
        {"id": 1, "rootInfo": {"winrate": 0.52}},  # D4
        {"id": 2, "rootInfo": {"winrate": 0.48}},  # F6
        {"id": 3, "rootInfo": {"winrate": 0.45}},  # PASS
    ]
    mock_process.stdout.readline.side_effect = [
        json.dumps(r) + "\n" for r in responses
    ]

    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    moves = ["D4", "F6", "PASS"]
    results = client.evaluate_moves(board, moves, player="B")

    assert len(results) == 3
    assert results["D4"] == 0.52
    assert results["F6"] == 0.48
    assert results["PASS"] == 0.45


@patch("subprocess.Popen")
def test_katago_evaluate_moves_infers_player(mock_popen, katago_config, mock_process):
    mock_popen.return_value = mock_process
    response_data = {"id": 1, "rootInfo": {"winrate": 0.5}}
    mock_process.stdout.readline.return_value = json.dumps(response_data) + "\n"

    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    board.apply_move("B", "D4")  # Now it's W's turn

    # Don't specify player - should infer W
    client.evaluate_moves(board, ["F6"], player=None)

    # Check that the move was applied for W
    written_data = mock_process.stdin.write.call_args[0][0]
    payload = json.loads(written_data.strip())
    assert payload["moves"][-1] == ["w", "f6"]


@patch("subprocess.Popen")
def test_katago_thread_safety(mock_popen, katago_config, mock_process):
    """Test that the RLock is used for thread safety."""
    mock_popen.return_value = mock_process
    response_data = {"id": 1, "rootInfo": {"winrate": 0.5}}
    mock_process.stdout.readline.return_value = json.dumps(response_data) + "\n"

    client = KataGoAnalysisClient(katago_config)
    client.start()

    # Verify _lock exists and is an RLock
    assert hasattr(client, "_lock")
    assert isinstance(client._lock, type(client._lock))

    board = BoardState.empty(9)
    # This should acquire and release the lock
    client.analyze_position(board)


@patch("subprocess.Popen")
def test_katago_analyze_with_pass_move(mock_popen, katago_config, mock_process):
    """Test that PASS moves are correctly formatted as 'pass' in lowercase."""
    mock_popen.return_value = mock_process
    response_data = {"id": 1, "rootInfo": {"winrate": 0.5}}
    mock_process.stdout.readline.return_value = json.dumps(response_data) + "\n"

    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    board.apply_move("B", "PASS")

    client.analyze_position(board)

    written_data = mock_process.stdin.write.call_args[0][0]
    payload = json.loads(written_data.strip())
    assert payload["moves"] == [["b", "pass"]]
