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

    # Run tests for a specific environment
    python -m tests.performance_tests.main --environment brax/ant

    # Run tests for a specific environment suite
    python -m tests.performance_tests.main --env-suite brax

    # Establish new baseline
    python -m tests.performance_tests.main --establish-baseline

    # List available tests
    python -m tests.performance_tests.main --list

    # Load configuration overrides from a JSON file
    python -m tests.performance_tests.main --config path/to/config.json

    # Specify report output directory
    python -m tests.performance_tests.main --report-dir path/to/reports

    # Enable verbose logging
    python -m tests.performance_tests.main --verbose
    
    # Run each test with multiple seeds
    python -m tests.performance_tests.main --num-seeds 3
"""

import argparse
import json
import logging

from stoix.tests.performance_tests.framework.runner import (
    discover_tests,
    run_tests,
    generate_report,
    list_available_tests,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        "--env-suite",
        dest="env_suites",
        action="append",
        help="Environment suite to test (e.g., 'brax' for all brax environments)",
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
        default="stoix/tests/performance_tests/data/reports",
        help="Directory to save reports",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--num-seeds", 
        type=int, 
        default=1, 
        help="Number of seeds (runs) per test for statistical analysis"
    )

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
            
            # Organize tests by algorithm, env suite, and specific env
            algos_dict = {}
            for algo, env, module_path, arch in sorted(tests):
                # Split environment into suite and scenario
                if '/' in env:
                    env_suite, env_scenario = env.split('/', 1)
                else:
                    env_suite, env_scenario = 'other', env
                
                if algo not in algos_dict:
                    algos_dict[algo] = {}
                if env_suite not in algos_dict[algo]:
                    algos_dict[algo][env_suite] = {}
                if env_scenario not in algos_dict[algo][env_suite]:
                    algos_dict[algo][env_suite][env_scenario] = []
                algos_dict[algo][env_suite][env_scenario].append((module_path, arch))
            
            # Display tests in organized format
            for algo in sorted(algos_dict.keys()):
                print(f"\n📊 Algorithm: {algo}")
                for env_suite in sorted(algos_dict[algo].keys()):
                    print(f"  🌐 Suite: {env_suite}")
                    for env_scenario in sorted(algos_dict[algo][env_suite].keys()):
                        print(f"    🌍 Scenario: {env_scenario}")
                        for module_path, arch in algos_dict[algo][env_suite][env_scenario]:
                            print(f"      - {module_path} (arch: {arch})")
        return

    # Load config overrides from JSON file if specified
    config_overrides = {}
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
        env_suites=args.env_suites,
        establish_baseline=args.establish_baseline,
        config_overrides=config_overrides,
        num_seeds=args.num_seeds,
    )

    # Generate report
    if results:
        report_path = generate_report(results, args.report_dir)
        logger.info(f"Report generated and saved to {report_path}")
    else:
        logger.warning("No test results to report")


if __name__ == "__main__":
    main()
