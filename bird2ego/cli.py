"""Command-line entry point for the bird2ego pipeline."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bird2ego",
        description="Process a video with the industrial task analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bird2ego --input video.mp4
  bird2ego --config configs/real.yaml --input video.mp4 --out results/
  bird2ego -i video.mp4 -o output/ --run-id my_run_001
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: data/processed/<run_id>/)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (default: timestamp)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from command-line arguments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = build_parser().parse_args(argv)

    setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out or f"data/processed/{run_id}"

    logger.info("=" * 60)
    logger.info("bird2ego task video analysis pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)

    try:
        output_paths = run_pipeline(
            video_path=str(input_path),
            config_path=str(config_path),
            output_dir=output_dir,
            run_id=run_id,
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1

    logger.info("=" * 60)
    logger.info("Output files:")
    for name, path in output_paths.items():
        logger.info(f"  {name}: {path}")
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    return 0
