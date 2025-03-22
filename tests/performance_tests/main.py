#!/usr/bin/env python3
"""
Performance Test Runner for Stoix Reinforcement Learning Algorithms

This script provides a standardized framework for running performance tests on
reinforcement learning algorithms implemented in the Stoix library. It allows for:

1. Running tests for specific algorithm-environment combinations
2. Comparing performance against established baselines
3. Establishing new baselines for future comparisons
4. Generating detailed reports on test results

The system integrates with the existing Hydra configuration framework to avoid
duplicating configuration definitions, allowing direct reuse of the same configs
used in normal training.

Usage:
    # Run all tests
    python -m tests.performance_tests.main

    # Run tests for a specific algorithm
    python -m tests.performance_tests.main --algorithm ff_sac

    # Establish new baseline
    python -m tests.performance_tests.main --establish-baseline

    # Override configuration parameters
    python -m tests.performance_tests.main --config-override "system.learning_rate=1e-4"
"""

import os
import sys
import argparse
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import core functionality
from tests.performance_tests.framework.runner import (
    discover_tests,
    run_tests,
    generate_report,
    list_available_tests,
)


def main():
    """Main entry point for the performance testing script."""
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Run performance tests for reinforcement learning algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithms",
        action="append",
        help="Algorithm to test (can specify multiple)",
    )
    parser.add_argument(
        "--environment",
        dest="environments",
        action="append",
        help="Environment to test (can specify multiple)",
    )
    parser.add_argument(
        "--establish-baseline", action="store_true", help="Establish new baselines"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--config", type=str, help="Path to a JSON file with configuration overrides"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tests and exit"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="tests/performance_tests/data/reports",
        help="Directory to save reports",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("tests.performance_tests").setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger("tests.performance_tests").setLevel(logging.INFO)

    # Discover and register tests
    discover_tests()

    # Get available tests
    tests = list_available_tests()

    # Just list tests if requested
    if args.list:
        if not tests:
            print("No tests have been registered.")
        else:
            print("Available tests:")
            for algo, env in sorted(tests):
                print(f"  {algo} on {env}")
        return

    # Load config overrides from JSON file if specified
    config_overrides = None
    if args.config:
        try:
            with open(args.config, "r") as f:
                config_overrides = json.load(f)
            logger.info(f"Loaded config overrides from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return

    # Run tests
    results = run_tests(
        algorithms=args.algorithms,
        environments=args.environments,
        establish_baseline=args.establish_baseline,
        max_steps=args.max_steps,
        config_overrides=config_overrides,
    )

    # Generate report
    if results:
        report_path = generate_report(results, args.report_dir)
        logger.info(f"Report generated and saved to {report_path}")
    else:
        logger.warning("No test results to report")


if __name__ == "__main__":
    main()
