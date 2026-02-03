"""Self-play engine for generating training data from trained models.

This module provides functionality to generate game data by having a model
play against itself, useful for curriculum learning and data augmentation.
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .board import BoardState, coords_to_gtp
from .prompting import build_prompt, parse_move

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play game generation."""

    model_path: str
    board_size: int = 9
    num_games: int = 100
    max_moves: int = 200
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 10
    output_dir: str = "data/self_play"
    parallel_games: int = 1
    seed: Optional[int] = None


@dataclass
class GameResult:
    """Result of a self-play game."""

    board_size: int
    moves: List[Tuple[str, str]]  # (player, move) pairs
    move_count: int
    illegal_move_count: int
    pass_count: int
    completed: bool
    winner: Optional[str] = None
    score: Optional[float] = None


class SelfPlayEngine:
    """Engine for generating self-play games."""

    def __init__(self, config: SelfPlayConfig):
        """Initialize self-play engine.

        Args:
            config: Self-play configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._rng = random.Random(config.seed)

    def load_model(self) -> None:
        """Load the trained model for self-play."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading model from %s", self.config.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto",
                torch_dtype="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            logger.info("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def generate_move(self, board: BoardState, player: str) -> str:
        """Generate a move for the current position.

        Args:
            board: Current board state
            player: Player to move ('B' or 'W')

        Returns:
            Generated move string
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build prompt
        prompt = build_prompt(board, player)

        # Generate move
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )

            # Decode and extract move
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Parse move from response
            move = parse_move(response)
            return move

        except Exception as e:
            logger.debug(f"Error generating move: {e}")
            return "PASS"  # Fallback to pass on error

    def play_game(self, game_id: int) -> GameResult:
        """Play a single self-play game.

        Args:
            game_id: Game identifier for logging

        Returns:
            GameResult with game data
        """
        logger.info("Starting self-play game %d", game_id)

        board = BoardState(size=self.config.board_size)
        moves: List[Tuple[str, str]] = []
        illegal_moves = 0
        consecutive_passes = 0

        for move_num in range(self.config.max_moves):
            # Determine current player
            player = "B" if move_num % 2 == 0 else "W"

            # Generate move
            try:
                move = self.generate_move(board, player)
            except Exception as e:
                logger.warning(f"Failed to generate move: {e}")
                move = "PASS"

            # Track passes
            if move.upper() == "PASS":
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            # End game if both players pass
            if consecutive_passes >= 2:
                logger.info(
                    "Game %d ended after %d moves (both players passed)",
                    game_id,
                    move_num + 1,
                )
                return GameResult(
                    board_size=self.config.board_size,
                    moves=moves,
                    move_count=len(moves),
                    illegal_move_count=illegal_moves,
                    pass_count=sum(1 for _, m in moves if m.upper() == "PASS"),
                    completed=True,
                )

            # Try to apply move
            try:
                board.apply_move(player, move)
                moves.append((player, move))
            except Exception as e:
                logger.debug(f"Illegal move attempted: {move} by {player}: {e}")
                illegal_moves += 1

                # After too many illegal moves, end game
                if illegal_moves >= 10:
                    logger.warning(
                        "Game %d ended early (too many illegal moves)", game_id
                    )
                    return GameResult(
                        board_size=self.config.board_size,
                        moves=moves,
                        move_count=len(moves),
                        illegal_move_count=illegal_moves,
                        pass_count=sum(1 for _, m in moves if m.upper() == "PASS"),
                        completed=False,
                    )

        # Game reached max moves
        logger.info(
            "Game %d reached max moves (%d)", game_id, self.config.max_moves
        )
        return GameResult(
            board_size=self.config.board_size,
            moves=moves,
            move_count=len(moves),
            illegal_move_count=illegal_moves,
            pass_count=sum(1 for _, m in moves if m.upper() == "PASS"),
            completed=False,
        )

    def generate_games(self) -> List[GameResult]:
        """Generate multiple self-play games.

        Uses parallel processing if parallel_games > 1, otherwise sequential.

        Returns:
            List of game results
        """
        if self.model is None:
            self.load_model()

        if self.config.parallel_games > 1:
            return self._generate_games_parallel()
        else:
            return self._generate_games_sequential()

    def _generate_games_sequential(self) -> List[GameResult]:
        """Generate games sequentially (original implementation).

        Returns:
            List of game results
        """
        results: List[GameResult] = []

        for game_id in range(self.config.num_games):
            try:
                result = self.play_game(game_id)
                results.append(result)

                # Log progress
                if (game_id + 1) % 10 == 0:
                    avg_moves = sum(r.move_count for r in results) / len(results)
                    avg_illegal = sum(r.illegal_move_count for r in results) / len(
                        results
                    )
                    logger.info(
                        "Progress: %d/%d games | Avg moves: %.1f | Avg illegal: %.1f",
                        game_id + 1,
                        self.config.num_games,
                        avg_moves,
                        avg_illegal,
                    )

            except Exception as e:
                logger.error(f"Error in game {game_id}: {e}")
                continue

        return results

    def _generate_games_parallel(self) -> List[GameResult]:
        """Generate games in parallel using threading.

        Uses ThreadPoolExecutor to run multiple games concurrently.
        Threading is used rather than multiprocessing because:
        1. The model is GPU-bound and benefits from shared GPU access
        2. GIL is released during model inference
        3. Avoids overhead of pickling large models

        Returns:
            List of game results
        """
        logger.info(
            "Generating %d games in parallel (max %d concurrent)",
            self.config.num_games,
            self.config.parallel_games,
        )

        results: List[GameResult] = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.config.parallel_games) as executor:
            # Submit all games
            future_to_game_id = {
                executor.submit(self.play_game, game_id): game_id
                for game_id in range(self.config.num_games)
            }

            # Collect results as they complete
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    # Log progress
                    if completed_count % 10 == 0:
                        avg_moves = sum(r.move_count for r in results) / len(results)
                        avg_illegal = sum(r.illegal_move_count for r in results) / len(
                            results
                        )
                        logger.info(
                            "Progress: %d/%d games | Avg moves: %.1f | Avg illegal: %.1f",
                            completed_count,
                            self.config.num_games,
                            avg_moves,
                            avg_illegal,
                        )

                except Exception as e:
                    logger.error(f"Error in game {game_id}: {e}")
                    continue

        # Sort results by original game order for consistency
        logger.info(
            "Completed %d/%d games in parallel",
            len(results),
            self.config.num_games,
        )
        return results

    def save_games_sgf(self, results: List[GameResult], output_dir: Path) -> None:
        """Save games to SGF files.

        Args:
            results: List of game results
            output_dir: Directory to save SGF files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(results):
            sgf_path = output_dir / f"self_play_{i:05d}.sgf"

            # Build SGF content
            sgf_content = self._build_sgf(result, i)

            # Write to file
            with open(sgf_path, "w", encoding="utf-8") as f:
                f.write(sgf_content)

        logger.info("Saved %d SGF files to %s", len(results), output_dir)

    def _build_sgf(self, result: GameResult, game_id: int) -> str:
        """Build SGF content for a game result.

        Args:
            result: Game result to convert
            game_id: Game identifier

        Returns:
            SGF content string
        """
        # SGF header
        lines = [
            "(;GM[1]FF[4]CA[UTF-8]AP[Go RL Self-Play]",
            f"SZ[{result.board_size}]",
            f"KM[7.5]",
            f"PB[Model]PW[Model]",
            f"RE[?]",  # Result unknown without scoring
            f"DT[{game_id}]",
        ]

        # Add moves
        for player, move in result.moves:
            if move.upper() == "PASS":
                move_sgf = ""
            else:
                # Convert GTP to SGF coordinates
                try:
                    from .board import gtp_to_coords

                    row, col = gtp_to_coords(move, result.board_size)
                    # SGF coordinates use (col, row) with origin at bottom-left.
                    sgf_x = chr(ord("a") + col)
                    sgf_y = chr(ord("a") + (result.board_size - 1 - row))
                    move_sgf = f"{sgf_x}{sgf_y}"
                except Exception:
                    continue  # Skip invalid moves

            player_tag = "B" if player == "B" else "W"
            lines.append(f";{player_tag}[{move_sgf}]")

        lines.append(")")

        return "\n".join(lines)


def compute_game_statistics(results: List[GameResult]) -> dict:
    """Compute statistics for a set of self-play games.

    Args:
        results: List of game results

    Returns:
        Dictionary of statistics
    """
    if not results:
        return {}

    total_games = len(results)
    completed_games = sum(1 for r in results if r.completed)
    total_moves = sum(r.move_count for r in results)
    total_illegal = sum(r.illegal_move_count for r in results)
    total_passes = sum(r.pass_count for r in results)

    return {
        "total_games": total_games,
        "completed_games": completed_games,
        "completion_rate": completed_games / total_games if total_games > 0 else 0.0,
        "avg_moves_per_game": total_moves / total_games if total_games > 0 else 0.0,
        "avg_illegal_per_game": total_illegal / total_games if total_games > 0 else 0.0,
        "avg_passes_per_game": total_passes / total_games if total_games > 0 else 0.0,
        "total_moves": total_moves,
        "total_illegal": total_illegal,
        "total_passes": total_passes,
    }
