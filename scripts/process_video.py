#!/usr/bin/env python
"""CLI script for processing videos with the vision pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


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


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Process a video with the industrial task analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_video.py --input video.mp4
  python process_video.py --config custom.yaml --input video.mp4 --out results/
  python process_video.py -i video.mp4 -o output/ --run-id my_run_001
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input video file",
    )

    parser.add_argument(
        "--out", "-o",
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Determine output directory
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        output_dir = args.out
    else:
        output_dir = f"data/processed/{run_id}"

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to script, then relative to cwd
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / args.config
        if not config_path.exists():
            config_path = Path(args.config)

    logger.info("=" * 60)
    logger.info("Industrial Task Video Analysis Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)

    try:
        # Run pipeline
        output_paths = run_pipeline(
            video_path=str(input_path),
            config_path=str(config_path),
            output_dir=output_dir,
            run_id=run_id,
        )

        # Print output paths
        logger.info("=" * 60)
        logger.info("Output files:")
        for name, path in output_paths.items():
            logger.info(f"  {name}: {path}")
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
