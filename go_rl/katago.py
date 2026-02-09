from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from time import sleep
from typing import Dict, Iterable, List, Optional

from .board import BoardState, coords_to_gtp, gtp_to_coords
from .config import KataGoConfig

logger = logging.getLogger(__name__)


@dataclass
class KataGoResult:
    winrate: float
    raw: Dict[str, object]
    score_lead: Optional[float] = None
    best_move: Optional[str] = None


@dataclass
class _PendingRequest:
    event: threading.Event
    response: Optional[Dict[str, object]] = None
    error: Optional[BaseException] = None


class KataGoAnalysisClient:
    def __init__(self, config: KataGoConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None
        self._write_lock = threading.RLock()
        self._lock = self._write_lock
        self._pending_lock = threading.RLock()
        self._pending: Dict[int, _PendingRequest] = {}
        self._shutdown_event = threading.Event()
        self._next_id = 1
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._process is not None:
            logger.debug("KataGo process already running")
            return
        cmd = [
            self.config.binary_path,
            "analysis",
            "-model",
            self.config.model_path,
            "-config",
            self.config.config_path,
        ]
        env = os.environ.copy()
        if self.config.katago_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.katago_cuda_visible_devices
        logger.info("Starting KataGo analysis process")
        self._shutdown_event.clear()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if self._process.stdout is not None:
            self._stdout_thread = threading.Thread(
                target=self._drain_stdout, daemon=True
            )
            self._stdout_thread.start()
        if self._process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stderr_thread.start()

    def close(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping KataGo analysis process")
        process = self._process
        self._process = None
        self._shutdown_event.set()
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        self._fail_all_pending(RuntimeError("KataGo process closed."))
        if self._stdout_thread:
            self._stdout_thread.join(timeout=1)
            self._stdout_thread = None
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while not self._shutdown_event.is_set():
            line = process.stderr.readline()
            if line == "":
                break
            logger.debug("KataGo stderr: %s", line.rstrip())

    def _drain_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while not self._shutdown_event.is_set():
            with self._pending_lock:
                has_pending = bool(self._pending)
            if not has_pending:
                sleep(0.001)
                continue
            line = process.stdout.readline()
            if line == "":
                if not self._shutdown_event.is_set():
                    self._fail_all_pending(
                        RuntimeError(
                            "KataGo process ended before all responses were received."
                        )
                    )
                return
            text = line.strip()
            if not text:
                continue
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Failed to decode KataGo response line: %r", text)
                continue
            req_id = response.get("id")
            if not isinstance(req_id, int):
                logger.warning("KataGo response missing integer id: %r", response)
                continue
            with self._pending_lock:
                pending = self._pending.get(req_id)
            if pending is None:
                logger.debug("Ignoring response for unknown request id=%s", req_id)
                continue
            pending.response = response
            pending.event.set()

    def _fail_all_pending(self, error: BaseException) -> None:
        with self._pending_lock:
            pending_entries = list(self._pending.values())
            self._pending.clear()
        for pending in pending_entries:
            pending.error = error
            pending.event.set()

    def _next_request_id(self) -> int:
        with self._write_lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _register_pending(self, req_id: int) -> _PendingRequest:
        pending = _PendingRequest(event=threading.Event())
        with self._pending_lock:
            if req_id in self._pending:
                raise RuntimeError(f"Duplicate request id: {req_id}")
            self._pending[req_id] = pending
        return pending

    def _write_payload(self, payload: Dict[str, object]) -> None:
        if self._process is None:
            raise RuntimeError("KataGo process not started.")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("KataGo process streams are unavailable.")
        logger.debug("Sending KataGo request id=%s", payload.get("id"))
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

    def _submit_payload(self, payload: Dict[str, object]) -> int:
        req_id = payload.get("id")
        if not isinstance(req_id, int):
            raise ValueError("KataGo request payload requires an integer id.")
        pending = self._register_pending(req_id)
        try:
            with self._write_lock:
                self._write_payload(payload)
        except Exception as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            pending.error = exc
            pending.event.set()
            raise
        return req_id

    def _await_raw_result(
        self,
        req_id: int,
        timeout: Optional[float] = None,
    ) -> Dict[str, object]:
        with self._pending_lock:
            pending = self._pending.get(req_id)
        if pending is None:
            raise RuntimeError(f"Unknown request id: {req_id}")
        wait_timeout = (
            timeout if timeout is not None else self.config.request_timeout_seconds
        )
        if not pending.event.wait(wait_timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(f"KataGo request {req_id} timed out.")
        with self._pending_lock:
            self._pending.pop(req_id, None)
        if pending.error is not None:
            raise RuntimeError(f"KataGo request {req_id} failed.") from pending.error
        if pending.response is None:
            raise RuntimeError("KataGo returned empty response.")
        logger.debug("Received KataGo response id=%s", req_id)
        return pending.response

    def _send(self, payload: Dict[str, object]) -> Dict[str, object]:
        if self._process is None:
            raise RuntimeError("KataGo process not started.")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("KataGo process streams are unavailable.")
        req_id = self._submit_payload(payload)
        return self._await_raw_result(req_id)

    def _build_payload(self, board: BoardState, req_id: int) -> Dict[str, object]:
        return {
            "id": req_id,
            "boardXSize": board.size,
            "boardYSize": board.size,
            "rules": self.config.rules,
            "komi": self.config.komi,
            "moves": [
                [player.lower(), coords_to_gtp(coords, board.size).lower()]
                if coords is not None
                else [player.lower(), "pass"]
                for player, coords in board.moves
            ],
        }

    def submit_position(self, board: BoardState) -> int:
        req_id = self._next_request_id()
        payload = self._build_payload(board, req_id)
        logger.debug(
            "Submitting position id=%d with %d moves",
            req_id,
            len(board.moves),
        )
        self._submit_payload(payload)
        return req_id

    def await_result(
        self,
        req_id: int,
        timeout: Optional[float] = None,
    ) -> KataGoResult:
        result = self._await_raw_result(req_id, timeout=timeout)
        return KataGoResult(
            winrate=_extract_winrate(result),
            raw=result,
            score_lead=_extract_score_lead(result),
            best_move=_extract_best_move(result),
        )

    def analyze_position(self, board: BoardState) -> KataGoResult:
        req_id = self.submit_position(board)
        return self.await_result(req_id)

    def analyze_positions(self, boards: Iterable[BoardState]) -> List[KataGoResult]:
        board_list = list(boards)
        if not board_list:
            return []
        request_ids = [self.submit_position(board) for board in board_list]
        return [self.await_result(req_id) for req_id in request_ids]

    def evaluate_moves(
        self,
        board: BoardState,
        moves: Iterable[str],
        player: Optional[str] = None,
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        move_player = player.strip().upper() if player else board.next_player()
        move_list = list(moves)
        logger.debug(
            "Evaluating %d moves for player %s",
            len(move_list),
            move_player,
        )
        candidate_boards: List[BoardState] = []
        for move in move_list:
            candidate = BoardState(size=board.size, moves=list(board.moves))
            candidate.apply_move(move_player, move)
            candidate_boards.append(candidate)
        analyses = self.analyze_positions(candidate_boards)
        for move, analysis in zip(move_list, analyses):
            results[move] = analysis.winrate
        return results

    def choose_move(self, board: BoardState, player: Optional[str] = None) -> str:
        """Choose KataGo's best move for the provided board and player.

        Falls back to PASS when no move is available or the suggested move is invalid.
        """
        move_player = player.strip().upper() if player else board.next_player()
        analysis_board = BoardState(size=board.size, moves=list(board.moves))
        if analysis_board.next_player() != move_player:
            analysis_board.apply_move(analysis_board.next_player(), "PASS", strict=True)

        analysis = self.analyze_position(analysis_board)
        best_move = analysis.best_move
        if not best_move:
            return "PASS"
        try:
            coords = gtp_to_coords(best_move, board.size)
        except Exception:
            return "PASS"
        if not analysis_board.is_strict_legal(move_player, coords):
            return "PASS"
        return best_move


def _extract_winrate(result: Dict[str, object]) -> float:
    root_info = result.get("rootInfo")
    if isinstance(root_info, dict) and "winrate" in root_info:
        return float(root_info["winrate"])
    raise KeyError("KataGo response missing rootInfo.winrate")


def _extract_score_lead(result: Dict[str, object]) -> Optional[float]:
    root_info = result.get("rootInfo")
    if isinstance(root_info, dict) and "scoreLead" in root_info:
        return float(root_info["scoreLead"])
    return None


def _extract_best_move(result: Dict[str, object]) -> Optional[str]:
    move_infos = result.get("moveInfos")
    if not isinstance(move_infos, list) or not move_infos:
        return None
    best = move_infos[0]
    if not isinstance(best, dict):
        return None
    move = best.get("move")
    if not isinstance(move, str) or not move.strip():
        return None
    normalized = move.strip().upper()
    if normalized == "PASS":
        return "PASS"
    return normalized
