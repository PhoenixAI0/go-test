"""Referee GTP games between a model engine and KataGo.

KataGo's own `match` command only plays KataGo networks, so evaluation drives two
GTP engines here. Moves are checked with the repository's strict rules (suicide
and simple ko are illegal). An illegal move or an exhausted clock loses, and the
trusted KataGo engine scores games that end by passing. A move cap truncates
a game and is recorded as Void, never as a scored win.
"""
from __future__ import annotations

import json
import math
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Optional

from .board import BoardState, gtp_to_coords
from .game_result import parse_game_result
from .rules import legality_superko

# Process scheduling and pipe latency are not charged to a player's clock.
_GRACE_SECONDS = 10.0
# Loading a network can take minutes on a CPU.
_SETUP_TIMEOUT_SECONDS = 600.0


class GtpError(RuntimeError):
    """The engine answered a command with a GTP failure response."""


@dataclass
class PlayerConfig:
    name: str
    command: str


@dataclass
class MatchConfig:
    board_size: int
    num_games: int
    model: PlayerConfig
    katago: PlayerConfig
    komi: float = 7.5
    rules: str = "chinese"
    main_time: float = 60.0
    byoyomi_time: float = 5.0
    byoyomi_periods: int = 5
    sgf_output_dir: str = "match_sgfs"
    max_moves: Optional[int] = None

    def __post_init__(self) -> None:
        if self.board_size not in (9, 19):
            raise ValueError("board_size must be 9 or 19")
        legality_superko(self.rules)  # Refuse rules the local referee cannot faithfully enforce.
        if self.num_games < 1:
            raise ValueError("num_games must be positive")
        if self.main_time < 0 or self.byoyomi_time < 0 or self.byoyomi_periods < 0:
            raise ValueError("Time controls must be nonnegative")
        if self.main_time + self.byoyomi_time * self.byoyomi_periods <= 0:
            raise ValueError("Time controls must allow some thinking time")
        if not self.model.command.strip() or not self.katago.command.strip():
            raise ValueError("Both players need a GTP command")
        if self.model.name == self.katago.name:
            raise ValueError("Players need distinct names to attribute results")
        if self.max_moves is not None and self.max_moves < 1:
            raise ValueError("max_moves must be positive")
        if self.max_moves is None:
            # Generous for any finished game; stops two engines that never pass.
            self.max_moves = 3 * self.board_size * self.board_size

    @classmethod
    def load(cls, path: str | Path) -> "MatchConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["model"] = PlayerConfig(**data["model"])
        data["katago"] = PlayerConfig(**data["katago"])
        return cls(**data)


@dataclass
class _Clock:
    """Japanese byo-yomi: main time, then periods that reset after each move."""

    main: float
    byoyomi: float
    periods: int

    def deadline(self) -> float:
        return self.main + self.byoyomi * self.periods

    def spend(self, seconds: float) -> bool:
        """Charge a move; return False when the player has run out of time."""
        used = min(self.main, seconds)
        self.main -= used
        seconds -= used
        if seconds <= 0:
            return True
        if self.byoyomi <= 0:
            return False
        # A move finished inside a period keeps it; each exhausted period is lost.
        self.periods -= math.ceil(seconds / self.byoyomi) - 1
        return self.periods > 0


class GtpEngine:
    def __init__(self, name: str, command: str, stderr: IO[str] | None = None) -> None:
        self.name = name
        self._process = subprocess.Popen(
            shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        # A reader thread gives every command a deadline, even if the engine hangs.
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        self._closed = False

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line.rstrip("\r\n"))
        self._lines.put(None)

    def _next_line(self, deadline: float) -> str:
        try:
            line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError(f"{self.name} did not respond in time") from None
        if line is None:
            raise RuntimeError(f"{self.name} exited unexpectedly (exit code {self._process.poll()})")
        return line

    def send(self, command: str, timeout: float = _SETUP_TIMEOUT_SECONDS) -> str:
        """Send one command and return its response text; raise GtpError on '?'."""
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError(f"{self.name} exited unexpectedly") from None
        deadline = time.monotonic() + timeout
        line = self._next_line(deadline)
        while not line.strip():
            line = self._next_line(deadline)
        status = line[0]
        if status not in "=?":
            raise RuntimeError(f"{self.name} sent a malformed response to {command!r}: {line!r}")
        lines = [line[1:]]
        # A response ends with an empty line.
        while line := self._next_line(deadline):
            lines.append(line)
        text = "\n".join(lines).strip()
        if status == "?":
            raise GtpError(f"{self.name} rejected {command!r}: {text}")
        return text

    def try_send(self, command: str) -> None:
        """Send an optional command that not every engine supports."""
        try:
            self.send(command)
        except GtpError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.send("quit", timeout=5)
        except (RuntimeError, TimeoutError):
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        # A dead engine leaves unsent input buffered; drop it instead of raising later.
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        self._reader_thread.join(timeout=1)
        if self._process.stdout is not None:
            self._process.stdout.close()


@dataclass
class GameRecord:
    black: str
    white: str
    moves: list[tuple[str, str]]
    result: str
    termination: str = "scored"


def _other(color: str) -> str:
    return "W" if color == "B" else "B"


def play_game(
    black: GtpEngine,
    white: GtpEngine,
    scorer: GtpEngine,
    config: MatchConfig,
    log: Callable[[str], None] = lambda _message: None,
) -> GameRecord:
    """Play one game. `scorer` is trusted to judge move legality and the final score."""
    players = {"B": black, "W": white}
    for engine in players.values():
        engine.send(f"boardsize {config.board_size}")
        engine.send("clear_board")
        engine.send(f"komi {config.komi}")
        engine.send(f"kata-set-rules {config.rules}")
        engine.try_send(f"kgs-time_settings byoyomi {int(config.main_time)} "
                        f"{int(config.byoyomi_time)} {config.byoyomi_periods}")
    clocks = {color: _Clock(config.main_time, config.byoyomi_time, config.byoyomi_periods)
              for color in players}
    board = BoardState.empty(config.board_size)
    moves: list[tuple[str, str]] = []
    passes = 0
    game_started = time.monotonic()

    def finish(result: str, termination: str = "scored") -> GameRecord:
        log(f"{black.name} (B) vs {white.name} (W): {result} after {len(moves)} moves ({termination})")
        return GameRecord(black.name, white.name, moves, result, termination)

    for ply in range(config.max_moves):
        color = "B" if ply % 2 == 0 else "W"
        mover, opponent, clock = players[color], players[_other(color)], clocks[color]
        if clock.main > 0:
            mover.try_send(f"time_left {color.lower()} {int(clock.main)} 0")
        else:
            mover.try_send(f"time_left {color.lower()} {int(clock.byoyomi)} {clock.periods}")
        started = time.monotonic()
        try:
            reply = mover.send(f"genmove {color.lower()}", timeout=clock.deadline() + _GRACE_SECONDS)
        except TimeoutError:
            return finish(f"{_other(color)}+T", "timeout")
        except GtpError as exc:
            log(str(exc))
            return finish(f"{_other(color)}+F", "generation_error")
        if not clock.spend(time.monotonic() - started):
            return finish(f"{_other(color)}+T", "timeout")
        move = reply.strip().upper()
        if move == "RESIGN":
            return finish(f"{_other(color)}+R", "resign")
        try:
            board.apply_move(color, move, strict=True)
        except ValueError:
            log(f"{mover.name} played illegal move {reply!r}")
            return finish(f"{_other(color)}+F", "illegal_move")
        try:
            opponent.send(f"play {color.lower()} {move}")
        except GtpError:
            if opponent is not scorer:
                raise RuntimeError(f"{opponent.name} rejected legal move {color} {move}") from None
            # The trusted engine's rules (e.g. superko) can forbid what simple ko allows.
            log(f"{scorer.name} rejected {mover.name}'s move {move}")
            return finish(f"{_other(color)}+F", "referee_rejection")
        moves.append((color, move))
        if len(moves) % 50 == 0:
            log(f"{black.name} (B) vs {white.name} (W): {len(moves)} moves, "
                f"last {color} {move}, {time.monotonic() - game_started:.1f}s elapsed")
        passes = passes + 1 if move == "PASS" else 0
        if passes == 2:
            result = scorer.send("final_score")
            try:
                normalized, _ = parse_game_result(result)
            except ValueError:
                log(f"Referee returned an unscored final result: {result!r}")
                return finish("Void", "scoring_error")
            return finish(normalized)
    return finish("Void", "move_cap")


def _sgf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("]", "\\]")


def game_to_sgf(record: GameRecord, config: MatchConfig) -> str:
    nodes = []
    for color, move in record.moves:
        coords = gtp_to_coords(move, config.board_size)
        point = "" if coords is None else chr(ord("a") + coords[1]) + chr(ord("a") + coords[0])
        nodes.append(f";{color}[{point}]")
    comment = f"C[termination={_sgf_text(record.termination)}]" if record.termination != "scored" else ""
    return (f"(;GM[1]FF[4]CA[UTF-8]SZ[{config.board_size}]KM[{config.komi}]"
            f"RU[{_sgf_text(config.rules)}]PB[{_sgf_text(record.black)}]"
            f"PW[{_sgf_text(record.white)}]RE[{_sgf_text(record.result)}]{comment}{''.join(nodes)})\n")


def run_match(config: MatchConfig, sgf_dir: Path, log_file: Path) -> list[Path]:
    """Play every game, alternating colors, and write one SGF per game."""
    sgf_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Unique names keep earlier matches in the same directory intact.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    written: list[Path] = []
    with log_file.open("a", encoding="utf-8") as log_handle:
        def log(message: str) -> None:
            log_handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            log_handle.flush()
            print(message, flush=True)

        model = GtpEngine(config.model.name, config.model.command, stderr=log_handle)
        try:
            katago = GtpEngine(config.katago.name, config.katago.command, stderr=log_handle)
            try:
                for index in range(config.num_games):
                    black, white = (model, katago) if index % 2 == 0 else (katago, model)
                    record = play_game(black, white, katago, config, log)
                    path = sgf_dir / f"{stamp}_{index:04d}.sgf"
                    path.write_text(game_to_sgf(record, config), encoding="utf-8")
                    written.append(path)
            finally:
                katago.close()
        finally:
            model.close()
    return written
