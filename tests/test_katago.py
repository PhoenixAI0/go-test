import json
import queue
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from go_rl.board import BoardState
from go_rl.config import KataGoConfig
from go_rl.katago import (
    KataGoAnalysisClient,
    KataGoResult,
    _extract_best_move,
    _extract_score_lead,
    _extract_winrate,
)


class _QueueStdout:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._closed = False

    def readline(self) -> str:
        if self._closed and self._queue.empty():
            return ""
        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            return ""

    def push(self, line: str) -> None:
        self._queue.put(line)

    def close(self) -> None:
        self._closed = True
        self.push("")


class _EofStream:
    def readline(self) -> str:
        return ""

    def close(self) -> None:
        return None


class _FakeStdin:
    def __init__(self, on_write) -> None:
        self._on_write = on_write
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)
        self._on_write(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, on_request) -> None:
        self.stdout = _QueueStdout()
        self.stderr = _EofStream()
        self.stdin = _FakeStdin(self._handle_write)
        self._on_request = on_request
        self.terminate_called = False
        self.kill_called = False

    def _handle_write(self, data: str) -> None:
        payload = json.loads(data.strip())
        self._on_request(payload, self)

    def terminate(self) -> None:
        self.terminate_called = True
        self.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.kill_called = True
        self.stdout.close()


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


def test_extract_score_lead_success():
    response = {"rootInfo": {"scoreLead": 3.5}}
    assert _extract_score_lead(response) == 3.5


def test_extract_score_lead_missing():
    response = {"rootInfo": {"winrate": 0.55}}
    assert _extract_score_lead(response) is None


def test_extract_best_move_success():
    response = {"moveInfos": [{"move": "q16"}]}
    assert _extract_best_move(response) == "Q16"


def test_extract_best_move_missing():
    response = {"moveInfos": []}
    assert _extract_best_move(response) is None


def test_katago_client_init(katago_config):
    client = KataGoAnalysisClient(katago_config)
    assert client.config == katago_config
    assert client._process is None
    assert client._next_id == 1
    assert client._stderr_thread is None
    assert client._stdout_thread is None


@patch("subprocess.Popen")
def test_katago_start_process_with_cuda_override(mock_popen, katago_config):
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stdout.readline.return_value = ""
    process.stderr = MagicMock()
    process.stderr.readline.return_value = ""
    process.terminate = MagicMock()
    process.wait = MagicMock()
    process.kill = MagicMock()
    mock_popen.return_value = process

    katago_config.katago_cuda_visible_devices = "1"
    client = KataGoAnalysisClient(katago_config)
    client.start()

    assert mock_popen.call_count == 1
    args = mock_popen.call_args.args[0]
    kwargs = mock_popen.call_args.kwargs
    assert args == [
        "/usr/bin/katago",
        "analysis",
        "-model",
        "/models/kata.bin.gz",
        "-config",
        "/config/gtp.cfg",
        "-override-config",
        "reportAnalysisWinratesAs=BLACK",
    ]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert client._stdout_thread is not None
    client.close()


@patch("subprocess.Popen")
def test_katago_start_already_running(mock_popen, katago_config):
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stdout.readline.return_value = ""
    process.stderr = MagicMock()
    process.stderr.readline.return_value = ""
    process.terminate = MagicMock()
    process.wait = MagicMock()
    process.kill = MagicMock()
    mock_popen.return_value = process

    client = KataGoAnalysisClient(katago_config)
    client.start()
    first_process = client._process
    client.start()
    assert client._process is first_process
    assert mock_popen.call_count == 1
    client.close()


def test_katago_send_not_started(katago_config):
    client = KataGoAnalysisClient(katago_config)
    with pytest.raises(RuntimeError, match="KataGo process not started"):
        client._send({"id": 1})


@patch("subprocess.Popen")
def test_katago_send_success(mock_popen, katago_config):
    def on_request(payload, proc):
        proc.stdout.push(json.dumps({"id": payload["id"], "rootInfo": {"winrate": 0.5}}) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    payload = {"id": 1, "boardXSize": 9, "boardYSize": 9}
    result = client._send(payload)
    assert result["id"] == "1"
    assert result["rootInfo"]["winrate"] == 0.5
    client.close()


@patch("subprocess.Popen")
def test_katago_out_of_order_responses_mapped_by_id(mock_popen, katago_config):
    seen_ids: list[int] = []

    def on_request(payload, proc):
        req_id = int(payload["id"])
        seen_ids.append(req_id)
        if len(seen_ids) == 1:
            return
        second = {"id": req_id, "rootInfo": {"winrate": 0.9}}
        first = {"id": seen_ids[0], "rootInfo": {"winrate": 0.1}}
        proc.stdout.push(json.dumps(second) + "\n")
        proc.stdout.push(json.dumps(first) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    board_a = BoardState.empty(9)
    board_b = BoardState.empty(9)
    req_a = client.submit_position(board_a)
    req_b = client.submit_position(board_b)

    result_b = client.await_result(req_b, timeout=1.0)
    result_a = client.await_result(req_a, timeout=1.0)
    assert result_b.winrate == 0.9
    assert result_a.winrate == 0.1
    client.close()


@patch("subprocess.Popen")
def test_katago_timeout_path(mock_popen, katago_config):
    started = Event()

    def on_request(_payload, _proc):
        started.set()
        return

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    katago_config.request_timeout_seconds = 0.05
    client = KataGoAnalysisClient(katago_config)
    client.start()

    req_id = client.submit_position(BoardState.empty(9))
    assert started.is_set() is True
    with pytest.raises(TimeoutError, match="timed out"):
        client.await_result(req_id)
    client.close()


@patch("subprocess.Popen")
def test_katago_close_with_pending_requests(mock_popen, katago_config):
    def on_request(_payload, _proc):
        return

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    req_id = client.submit_position(BoardState.empty(9))
    client.close()
    with pytest.raises(RuntimeError, match="(Unknown request id|failed)"):
        client.await_result(req_id, timeout=0.1)


@patch("subprocess.Popen")
def test_katago_analyze_position_backward_compatible(mock_popen, katago_config):
    def on_request(payload, proc):
        response = {
            "id": payload["id"],
            "rootInfo": {"winrate": 0.65, "scoreLead": 1.2},
            "moveInfos": [{"move": "D4"}],
        }
        proc.stdout.push(json.dumps(response) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    board.apply_move("B", "D4")
    board.apply_move("W", "PASS")
    result = client.analyze_position(board)

    assert isinstance(result, KataGoResult)
    assert result.winrate == 0.65
    assert result.score_lead == 1.2
    assert result.best_move == "D4"
    assert client._next_id == 2
    client.close()


@patch("subprocess.Popen")
def test_katago_analyze_positions_batch(mock_popen, katago_config):
    def on_request(payload, proc):
        req_id = int(payload["id"])
        proc.stdout.push(json.dumps({"id": req_id, "rootInfo": {"winrate": 0.4 + (0.1 * req_id)}}) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    boards = [BoardState.empty(9), BoardState.empty(9)]
    results = client.analyze_positions(boards)
    assert len(results) == 2
    assert results[0].winrate != results[1].winrate
    client.close()


@patch("subprocess.Popen")
def test_katago_evaluate_moves(mock_popen, katago_config):
    def on_request(payload, proc):
        req_id = int(payload["id"])
        proc.stdout.push(json.dumps({"id": req_id, "rootInfo": {"winrate": 0.5 + (0.01 * req_id)}}) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    moves = ["D4", "F6", "PASS"]
    results = client.evaluate_moves(board, moves, player="B")
    assert len(results) == 3
    assert all(move in results for move in moves)
    client.close()


@patch("subprocess.Popen")
def test_katago_analyze_with_pass_move_payload(mock_popen, katago_config):
    def on_request(payload, proc):
        proc.stdout.push(json.dumps({"id": payload["id"], "rootInfo": {"winrate": 0.5}}) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()

    board = BoardState.empty(9)
    board.apply_move("B", "PASS")
    client.analyze_position(board)
    payload = json.loads(fake_process.stdin.writes[-1].strip())
    assert payload["moves"] == [["b", "pass"]]
    client.close()


@patch("subprocess.Popen")
def test_katago_choose_move_uses_best(mock_popen, katago_config):
    def on_request(payload, proc):
        response = {
            "id": payload["id"],
            "rootInfo": {"winrate": 0.5, "scoreLead": 0.0},
            "moveInfos": [{"move": "D4"}],
        }
        proc.stdout.push(json.dumps(response) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()
    board = BoardState.empty(9)
    assert client.choose_move(board, player="B") == "D4"
    client.close()


@patch("subprocess.Popen")
def test_katago_choose_move_fallback_pass(mock_popen, katago_config):
    def on_request(payload, proc):
        response = {
            "id": payload["id"],
            "rootInfo": {"winrate": 0.5},
            "moveInfos": [],
        }
        proc.stdout.push(json.dumps(response) + "\n")

    fake_process = _FakeProcess(on_request=on_request)
    mock_popen.return_value = fake_process
    client = KataGoAnalysisClient(katago_config)
    client.start()
    board = BoardState.empty(9)
    assert client.choose_move(board, player="B") == "PASS"
    client.close()


@patch("subprocess.Popen")
def test_katago_sends_string_ids_and_reports_engine_errors(mock_popen, katago_config):
    def on_request(payload, proc):
        assert isinstance(payload["id"], str)
        proc.stdout.push(json.dumps({"id": payload["id"], "error": "illegal position"}) + "\n")

    mock_popen.return_value = _FakeProcess(on_request=on_request)
    client = KataGoAnalysisClient(katago_config)
    client.start()
    try:
        with pytest.raises(RuntimeError, match="illegal position"):
            client.analyze_position(BoardState.empty(9))
    finally:
        client.close()


@patch("subprocess.Popen")
def test_katago_waits_for_final_analysis(mock_popen, katago_config):
    def on_request(payload, proc):
        for interim, winrate in [(True, 0.1), (False, 0.8)]:
            proc.stdout.push(json.dumps({"id": payload["id"], "isDuringSearch": interim,
                                        "rootInfo": {"winrate": winrate}}) + "\n")

    mock_popen.return_value = _FakeProcess(on_request=on_request)
    client = KataGoAnalysisClient(katago_config)
    client.start()
    try:
        assert client.analyze_position(BoardState.empty(9)).winrate == 0.8
    finally:
        client.close()


def test_engine_exit_preserves_completed_results_and_failed_handles(katago_config):
    client = KataGoAnalysisClient(katago_config)
    done = client._register_pending(1)
    done.response = {"id": "1", "rootInfo": {"winrate": 0.7}}
    done.event.set()
    client._register_pending(2)
    client._fail_all_pending(RuntimeError("engine exited"))
    assert client._await_raw_result(1)["rootInfo"]["winrate"] == 0.7
    with pytest.raises(RuntimeError, match="engine exited"):
        client._await_raw_result(2)
    assert not client._pending
