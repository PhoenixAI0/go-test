#!/usr/bin/env python3
"""Generate self-play training data from a trained model.

This script uses a trained model to play games against itself, generating
SGF files that can be used for additional training data (curriculum learning).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from go_rl.config import load_config
from go_rl.self_play import SelfPlayConfig, SelfPlayEngine, compute_game_statistics
from venv_utils import ensure_virtualenv


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Generate self-play training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 games from a trained model
  python scripts/run_self_play.py \\
      --model outputs/grpo_9x9/final \\
      --output data/self_play/9x9 \\
      --num-games 100

  # Use a config file
  python scripts/run_self_play.py --config config/self_play_9x9.yaml

  # Generate with custom settings
  python scripts/run_self_play.py \\
      --model outputs/grpo_19x19/final \\
      --board-size 19 \\
      --num-games 500 \\
      --temperature 0.8

For more information, see docs/SELF_PLAY_GUIDE.md
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model",
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
        "--num-games",
        "-n",
        type=int,
        default=100,
        help="Number of games to generate (default: 100)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for SGF files (default: data/self_play)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without generating games",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config from file or arguments
    if args.config:
        logger.info(f"Loading config from {args.config}")
        try:
            config = load_config(args.config, SelfPlayConfig)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
    else:
        # Build config from arguments
        if not args.model:
            logger.error("--model is required when not using --config")
            parser.print_help()
            return 1

        config = SelfPlayConfig(
            model_path=args.model,
            board_size=args.board_size,
            num_games=args.num_games,
            max_moves=args.max_moves,
            temperature=args.temperature,
            top_p=args.top_p,
            output_dir=args.output or f"data/self_play/{args.board_size}x{args.board_size}",
            seed=args.seed,
        )

    # Override output dir if specified
    if args.output:
        config.output_dir = args.output

    # Print configuration
    logger.info("Self-play configuration:")
    logger.info(f"  Model: {config.model_path}")
    logger.info(f"  Board size: {config.board_size}x{config.board_size}")
    logger.info(f"  Number of games: {config.num_games}")
    logger.info(f"  Max moves per game: {config.max_moves}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Top-p: {config.top_p}")
    logger.info(f"  Output directory: {config.output_dir}")
    if config.seed is not None:
        logger.info(f"  Random seed: {config.seed}")

    if args.dry_run:
        logger.info("Dry run - exiting without generating games")
        return 0

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize self-play engine
    logger.info("Initializing self-play engine...")
    engine = SelfPlayEngine(config)

    try:
        # Load model
        logger.info("Loading model...")
        engine.load_model()

        # Generate games
        logger.info(f"Generating {config.num_games} games...")
        results = engine.generate_games()

        # Compute statistics
        stats = compute_game_statistics(results)
        logger.info("\nSelf-play statistics:")
        logger.info(f"  Total games: {stats['total_games']}")
        logger.info(f"  Completed games: {stats['completed_games']} ({stats['completion_rate']:.1%})")
        logger.info(f"  Avg moves per game: {stats['avg_moves_per_game']:.1f}")
        logger.info(f"  Avg illegal moves per game: {stats['avg_illegal_per_game']:.1f}")
        logger.info(f"  Avg passes per game: {stats['avg_passes_per_game']:.1f}")

        # Save to SGF
        logger.info(f"\nSaving games to {output_dir}...")
        engine.save_games_sgf(results, output_dir)

        logger.info("\nSelf-play complete!")
        logger.info(f"  Generated {len(results)} games")
        logger.info(f"  SGF files saved to: {output_dir}/")
        logger.info("\nNext steps:")
        logger.info("  1. (Optional) Filter games: python scripts/filter_self_play_games.py")
        logger.info("  2. Convert to training data: python scripts/build_sft_dataset.py")
        logger.info(f"     --sgf-dir {output_dir} --output data/sft_self_play.jsonl")

        return 0

    except KeyboardInterrupt:
        logger.info("\nSelf-play interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during self-play: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
