from __future__ import annotations

import threading
from dataclasses import replace
from typing import Iterable, List, Sequence

from .board import BoardState
from .config import KataGoConfig
from .katago import KataGoAnalysisClient, KataGoResult


class KataGoPool:
    def __init__(self, configs: Sequence[KataGoConfig]) -> None:
        if not configs:
            raise ValueError("KataGoPool requires at least one worker config.")
        self._clients = [KataGoAnalysisClient(cfg) for cfg in configs]
        self._dispatch_lock = threading.Lock()
        self._next_index = 0

    @classmethod
    def from_config(cls, config: KataGoConfig) -> "KataGoPool":
        configs = [replace(config) for _ in range(config.katago_pool_size)]
        return cls(configs)

    def start(self) -> None:
        for client in self._clients:
            client.start()

    def close(self) -> None:
        first_error: Exception | None = None
        for client in self._clients:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _next_client(self) -> KataGoAnalysisClient:
        with self._dispatch_lock:
            client = self._clients[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._clients)
            return client

    def analyze_position(self, board: BoardState) -> KataGoResult:
        return self._next_client().analyze_position(board)

    def analyze_positions(self, boards: Iterable[BoardState]) -> List[KataGoResult]:
        board_list = list(boards)
        if not board_list:
            return []
        requests: List[tuple[int, KataGoAnalysisClient, int]] = []
        for index, board in enumerate(board_list):
            client = self._next_client()
            req_id = client.submit_position(board)
            requests.append((index, client, req_id))
        ordered: List[KataGoResult | None] = [None] * len(board_list)
        for index, client, req_id in requests:
            ordered[index] = client.await_result(req_id)
        results: List[KataGoResult] = []
        for result in ordered:
            if result is None:
                raise RuntimeError("KataGo pool returned an incomplete result set.")
            results.append(result)
        return results

    def evaluate_moves(
        self,
        board: BoardState,
        moves: Iterable[str],
        player: str | None = None,
    ) -> dict[str, float]:
        move_player = player.strip().upper() if player else board.next_player()
        move_list = list(moves)
        candidates: List[BoardState] = []
        for move in move_list:
            candidate = BoardState(size=board.size, moves=list(board.moves))
            candidate.apply_move(move_player, move)
            candidates.append(candidate)
        analyses = self.analyze_positions(candidates)
        return {move: analysis.winrate for move, analysis in zip(move_list, analyses)}

    def choose_move(self, board: BoardState, player: str | None = None) -> str:
        return self._next_client().choose_move(board, player=player)


def build_katago_service(config: KataGoConfig) -> KataGoAnalysisClient | KataGoPool:
    if config.katago_pool_size > 1:
        return KataGoPool.from_config(config)
    return KataGoAnalysisClient(config)
