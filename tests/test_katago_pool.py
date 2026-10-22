from __future__ import annotations

from go_rl.board import BoardState
from go_rl.config import KataGoConfig
from go_rl.katago import KataGoResult
from go_rl.katago_pool import KataGoPool, build_katago_service


class _DummyClient:
    instances: list["_DummyClient"] = []

    def __init__(self, _config: KataGoConfig) -> None:
        self.__class__.instances.append(self)
        self.started = False
        self.closed = False
        self._next_id = 1
        self._submitted: dict[int, BoardState] = {}

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def submit_position(self, board: BoardState) -> int:
        req_id = self._next_id
        self._next_id += 1
        self._submitted[req_id] = board
        return req_id

    def await_result(self, req_id: int, timeout: float | None = None) -> KataGoResult:
        _ = timeout
        board = self._submitted[req_id]
        return KataGoResult(
            winrate=float(len(board.stones())),
            raw={"id": req_id},
        )

    def analyze_position(self, board: BoardState) -> KataGoResult:
        return KataGoResult(winrate=float(len(board.stones())), raw={"id": -1})

    def choose_move(self, board: BoardState, player: str | None = None) -> str:
        _ = board, player
        return "PASS"


def _base_config(pool_size: int) -> KataGoConfig:
    return KataGoConfig(
        binary_path="/usr/bin/katago",
        model_path="/models/kata.bin.gz",
        config_path="/config/gtp.cfg",
        katago_pool_size=pool_size,
    )


def test_build_katago_service_uses_pool_when_requested(monkeypatch):
    _DummyClient.instances = []
    monkeypatch.setattr("go_rl.katago_pool.KataGoAnalysisClient", _DummyClient)
    service = build_katago_service(_base_config(pool_size=2))
    assert isinstance(service, KataGoPool)
    assert len(_DummyClient.instances) == 2


def test_katago_pool_round_robin_and_order(monkeypatch):
    _DummyClient.instances = []
    monkeypatch.setattr("go_rl.katago_pool.KataGoAnalysisClient", _DummyClient)
    pool = KataGoPool.from_config(_base_config(pool_size=2))
    pool.start()

    board_a = BoardState.empty(9)
    board_b = BoardState.empty(9)
    board_b.apply_move("B", "D4")
    board_c = BoardState.empty(9)
    board_c.apply_move("B", "D4")
    board_c.apply_move("W", "E5")

    results = pool.analyze_positions([board_a, board_b, board_c])
    assert [result.winrate for result in results] == [0.0, 1.0, 2.0]

    assert len(_DummyClient.instances[0]._submitted) == 2
    assert len(_DummyClient.instances[1]._submitted) == 1

    pool.close()
    assert _DummyClient.instances[0].closed is True
    assert _DummyClient.instances[1].closed is True


def test_pool_forwards_per_position_teacher_options(monkeypatch):
    calls = []

    class OptionClient(_DummyClient):
        def submit_position(self, board, **kwargs):
            calls.append(kwargs)
            return super().submit_position(board)

        def choose_move(self, board, player=None, **kwargs):
            calls.append(kwargs)
            return "PASS"

    monkeypatch.setattr("go_rl.katago_pool.KataGoAnalysisClient", OptionClient)
    pool = KataGoPool.from_config(_base_config(2))
    pool.analyze_positions([BoardState.empty(19)], max_visits=4, include_policy=True,
                           include_ownership=True, rules="chinese-ogs", komi=6.5)
    assert calls[0] == dict(max_visits=4, include_policy=True, include_ownership=True,
                            rules="chinese-ogs", komi=6.5)
    pool.choose_move(BoardState.empty(19), rules="chinese-ogs", komi=6.5)
    assert calls[1] == dict(rules="chinese-ogs", komi=6.5)


def test_pool_failed_batch_cancels_pending_workers(monkeypatch):
    import pytest
    cancelled = []

    class FailingClient(_DummyClient):
        def await_result(self, req_id, timeout=None):
            raise RuntimeError("engine failed")

        def cancel_request(self, req_id):
            cancelled.append((id(self), req_id))

    monkeypatch.setattr("go_rl.katago_pool.KataGoAnalysisClient", FailingClient)
    pool = KataGoPool.from_config(_base_config(2))
    with pytest.raises(RuntimeError, match="engine failed"):
        pool.analyze_positions([BoardState.empty(19), BoardState.empty(19)])
    assert len(cancelled) == 2
