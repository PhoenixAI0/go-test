#!/usr/bin/env python3
"""Interactive play against a trained Go model.

This script provides a simple text-based interface to play Go against
a trained model for testing and evaluation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.board import BoardState, coords_to_gtp
from go_rl.prompting import build_prompt, parse_move


def setup_logging(level: str = "WARNING") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s: %(message)s",
    )


def display_board(board: BoardState) -> None:
    """Display board in ASCII format.

    Args:
        board: Board state to display
    """
    size = board.size

    # Column labels
    col_labels = "  " + " ".join("ABCDEFGHJKLMNOPQRST"[:size])
    print("\n" + col_labels)

    # Board rows
    for y in range(size):
        row_label = f"{size - y:2d}"
        row = row_label + " "

        for x in range(size):
            stone = board.board[y][x]
            if stone == "B":
                row += "X "
            elif stone == "W":
                row += "O "
            else:
                # Show star points for 19x19
                if size == 19 and (x, y) in [(3, 3), (9, 3), (15, 3), (3, 9), (9, 9), (15, 9), (3, 15), (9, 15), (15, 15)]:
                    row += "+ "
                # Show star points for 9x9
                elif size == 9 and (x, y) in [(2, 2), (6, 2), (4, 4), (2, 6), (6, 6)]:
                    row += "+ "
                else:
                    row += ". "

        row += row_label
        print(row)

    print(col_labels + "\n")


def get_user_move(board: BoardState, player: str) -> Optional[str]:
    """Get move from user input.

    Args:
        board: Current board state
        player: Player color ('B' or 'W')

    Returns:
        Move string, "PASS", or None to quit
    """
    while True:
        try:
            user_input = input(f"{player} to play (e.g., D4, PASS, or Q to quit): ").strip().upper()

            if user_input in ["Q", "QUIT", "EXIT"]:
                return None

            if user_input in ["PASS", "P"]:
                return "PASS"

            if user_input in ["H", "HELP", "?"]:
                print("\nCommands:")
                print("  D4, Q16, etc. - Play a move")
                print("  PASS or P     - Pass")
                print("  Q or QUIT     - Quit game")
                print("  H or HELP     - Show this help")
                print()
                continue

            # Validate move format and legality
            # Try to apply it to check if it's legal
            test_board = BoardState(size=board.size, moves=list(board.moves))
            test_board.apply_move(player, user_input)

            return user_input

        except ValueError as e:
            print(f"Invalid move: {e}. Try again.")
        except KeyboardInterrupt:
            print("\nGame interrupted")
            return None
        except Exception as e:
            print(f"Error: {e}. Try again.")


def generate_model_move(model, tokenizer, board: BoardState, player: str, temperature: float = 0.8) -> str:
    """Generate move from model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        board: Current board state
        player: Player color ('B' or 'W')
        temperature: Sampling temperature

    Returns:
        Generated move string
    """
    try:
        # Build prompt
        prompt = build_prompt(board, player)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

        # Decode
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse move
        move = parse_move(response)
        return move

    except Exception as e:
        print(f"Model error: {e}")
        return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play Go interactively against a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play as Black against the model
  python scripts/play_interactive.py --model outputs/grpo_9x9/final

  # Play as White with custom settings
  python scripts/play_interactive.py \\
      --model outputs/grpo_19x19/final \\
      --board-size 19 \\
      --play-as white \\
      --temperature 0.5

Commands during play:
  D4, Q16, etc. - Play a move
  PASS or P     - Pass
  Q or QUIT     - Quit game
  H or HELP     - Show help
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=9,
        choices=[9, 13, 19],
        help="Board size (default: 9)",
    )
    parser.add_argument(
        "--play-as",
        choices=["black", "white"],
        default="black",
        help="Color to play as (default: black)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Model sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load model
    print("Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"Model loaded from {args.model}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Initialize game
    board = BoardState(size=args.board_size)
    user_color = "B" if args.play_as == "black" else "W"
    model_color = "W" if user_color == "B" else "B"

    print(f"\n{'=' * 50}")
    print(f"Interactive Go - {args.board_size}x{args.board_size}")
    print(f"{'=' * 50}")
    print(f"You play as: {user_color} ({'Black' if user_color == 'B' else 'White'})")
    print(f"Model plays: {model_color} ({'Black' if model_color == 'B' else 'White'})")
    print(f"Type 'H' for help, 'Q' to quit")
    print(f"{'=' * 50}\n")

    consecutive_passes = 0
    move_count = 0

    # Game loop
    while True:
        # Display board
        display_board(board)

        # Determine current player
        current_player = "B" if move_count % 2 == 0 else "W"

        # User's turn
        if current_player == user_color:
            move = get_user_move(board, current_player)

            if move is None:
                print("\nGame quit by user")
                break

            # Track passes
            if move.upper() == "PASS":
                consecutive_passes += 1
                print(f"{current_player} passes")
            else:
                consecutive_passes = 0
                print(f"{current_player} plays {move}")

            # Apply move
            try:
                board.apply_move(current_player, move)
            except Exception as e:
                print(f"Error applying move: {e}")
                continue

        # Model's turn
        else:
            print(f"{current_player} (Model) is thinking...")
            move = generate_model_move(model, tokenizer, board, current_player, args.temperature)

            # Track passes
            if move.upper() == "PASS":
                consecutive_passes += 1
                print(f"{current_player} passes")
            else:
                consecutive_passes = 0
                print(f"{current_player} plays {move}")

            # Apply move
            try:
                board.apply_move(current_player, move)
            except Exception as e:
                print(f"Model played illegal move: {move}")
                print(f"Error: {e}")
                print("Model forfeits")
                break

        # Check for game end (both players pass)
        if consecutive_passes >= 2:
            print("\n" + "=" * 50)
            print("Game ended (both players passed)")
            print("=" * 50)
            display_board(board)
            print(f"Total moves: {move_count + 1}")
            print("\nNote: Scoring not implemented. Use an SGF viewer to score the game.")
            break

        move_count += 1

        # Safety limit
        if move_count >= 500:
            print("\nMove limit reached (500 moves)")
            break

    print("\nThanks for playing!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
