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
        try:
            for client in self._clients:
                client.start()
        except Exception:
            # Do not leak earlier workers if a later engine cannot start.
            self.close()
            raise

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

    def analyze_position(
        self, board: BoardState, *, max_visits: int | None = None,
        include_policy: bool = False, include_ownership: bool = False,
        rules: str | dict[str, object] | None = None, komi: float | None = None,
    ) -> KataGoResult:
        options = _analysis_options(max_visits, include_policy, include_ownership, rules, komi)
        return self._next_client().analyze_position(board, **options)

    def analyze_positions(
        self, boards: Iterable[BoardState], *, max_visits: int | None = None,
        include_policy: bool = False, include_ownership: bool = False,
        rules: str | dict[str, object] | None = None, komi: float | None = None,
    ) -> List[KataGoResult]:
        board_list = list(boards)
        if not board_list:
            return []
        options = _analysis_options(max_visits, include_policy, include_ownership, rules, komi)
        requests: List[tuple[int, KataGoAnalysisClient, int]] = []
        try:
            for index, board in enumerate(board_list):
                client = self._next_client()
                req_id = client.submit_position(board, **options)
                requests.append((index, client, req_id))
            return [client.await_result(req_id) for _, client, req_id in requests]
        except BaseException:
            for _, client, req_id in requests:
                client.cancel_request(req_id)
            raise

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
            candidate = board.clone()
            candidate.apply_move(move_player, move, strict=True)
            candidates.append(candidate)
        analyses = self.analyze_positions(candidates)
        return {move: analysis.winrate for move, analysis in zip(move_list, analyses)}

    def choose_move(
        self, board: BoardState, player: str | None = None, *,
        rules: str | dict[str, object] | None = None, komi: float | None = None,
    ) -> str:
        options = _analysis_options(None, False, False, rules, komi)
        return self._next_client().choose_move(board, player=player, **options)


def _analysis_options(
    max_visits: int | None, include_policy: bool, include_ownership: bool,
    rules: str | dict[str, object] | None, komi: float | None,
) -> dict:
    options = {}
    if max_visits is not None:
        options["max_visits"] = max_visits
    if include_policy:
        options["include_policy"] = True
    if include_ownership:
        options["include_ownership"] = True
    if rules is not None:
        options["rules"] = rules
    if komi is not None:
        options["komi"] = komi
    return options


def build_katago_service(config: KataGoConfig) -> KataGoAnalysisClient | KataGoPool:
    if config.katago_pool_size > 1:
        return KataGoPool.from_config(config)
    return KataGoAnalysisClient(config)
