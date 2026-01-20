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
        self._lock = threading.Lock()
        self._next_id = 1

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

    def close(self) -> None:
        if self._process:
            logger.info("Stopping KataGo analysis process")
            self._process.terminate()
            self._process = None

    def _send(self, payload: Dict[str, object]) -> Dict[str, object]:
        if self._process is None:
            raise RuntimeError("KataGo process not started.")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("KataGo process streams are unavailable.")
        logger.debug("Sending KataGo request id=%s", payload.get("id"))
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline()
        if not response:
            raise RuntimeError("KataGo returned empty response.")
        logger.debug("Received KataGo response id=%s", payload.get("id"))
        return json.loads(response)

    def analyze_position(self, board: BoardState) -> KataGoResult:
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
