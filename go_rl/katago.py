from __future__ import annotations

import json
import logging
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .board import BoardState, coords_to_gtp
from .config import KataGoConfig

logger = logging.getLogger(__name__)


@dataclass
class KataGoResult:
    winrate: float
    raw: Dict[str, object]


class KataGoAnalysisClient:
    def __init__(self, config: KataGoConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.RLock()
        self._next_id = 1
        self._stderr_thread: Optional[threading.Thread] = None

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
        logger.info("Starting KataGo analysis process")
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stderr_thread.start()

    def close(self) -> None:
        if not self._process:
            return
        logger.info("Stopping KataGo analysis process")
        process = self._process
        self._process = None
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
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            logger.debug("KataGo stderr: %s", line.rstrip())

    def _send(self, payload: Dict[str, object]) -> Dict[str, object]:
        if self._process is None:
            raise RuntimeError("KataGo process not started.")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("KataGo process streams are unavailable.")
        with self._lock:
            logger.debug("Sending KataGo request id=%s", payload.get("id"))
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            response = self._process.stdout.readline()
            if not response:
                raise RuntimeError("KataGo returned empty response.")
            logger.debug("Received KataGo response id=%s", payload.get("id"))
            return json.loads(response)

    def analyze_position(self, board: BoardState) -> KataGoResult:
        with self._lock:
            payload = {
                "id": self._next_id,
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
            self._next_id += 1
            logger.debug(
                "Analyzing position id=%d with %d moves",
                payload["id"],
                len(board.moves),
            )
            result = self._send(payload)
        winrate = _extract_winrate(result)
        return KataGoResult(winrate=winrate, raw=result)

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
        for move in move_list:
            candidate = BoardState(size=board.size, moves=list(board.moves))
            candidate.apply_move(move_player, move)
            analysis = self.analyze_position(candidate)
            results[move] = analysis.winrate
        return results


def _extract_winrate(result: Dict[str, object]) -> float:
    root_info = result.get("rootInfo")
    if isinstance(root_info, dict) and "winrate" in root_info:
        return float(root_info["winrate"])
    raise KeyError("KataGo response missing rootInfo.winrate")
