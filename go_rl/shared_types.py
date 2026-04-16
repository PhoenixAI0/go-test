from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TYPE_CHECKING, TypeAlias, TypedDict

from .board import BoardState

if TYPE_CHECKING:
    from .katago import KataGoResult


BoardKey: TypeAlias = tuple[int, str, tuple[tuple[int, ...], ...]]


class KataGoRootInfo(TypedDict, total=False):
    winrate: float
    scoreLead: float


class KataGoMoveInfo(TypedDict, total=False):
    move: str


class KataGoResponse(TypedDict, total=False):
    id: int
    rootInfo: KataGoRootInfo
    moveInfos: list[KataGoMoveInfo]


class GpuInfo(TypedDict, total=False):
    available: bool
    backend: str
    device_count: int
    name: str
    total_memory_bytes: int
    torch_cuda: str | None
    torch_mps_built: bool
    torch_mps_available: bool
    detail: str


class EnvironmentInfo(TypedDict):
    python: str
    platform: str
    cwd: str
    packages: dict[str, str]
    gpu: GpuInfo


class RunLogPayload(TypedDict, total=False):
    run_name: str
    timestamp_utc: str
    config_path: str
    config: dict[str, object]
    environment: EnvironmentInfo
    extra: dict[str, object]


class KataGoService(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def analyze_position(self, board: BoardState) -> "KataGoResult": ...

    def analyze_positions(self, boards: Iterable[BoardState]) -> list["KataGoResult"]: ...

    def choose_move(self, board: BoardState, player: str | None = None) -> str: ...


def board_key(board: BoardState) -> BoardKey:
    return (
        board.size,
        board.next_player(),
        tuple(tuple(row) for row in board.current_board()),
    )
