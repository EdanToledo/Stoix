#!/usr/bin/env python3
"""
Performance Test Runner for Stoix Reinforcement Learning Algorithms

This module provides the core functionality for running performance tests,
including test discovery, execution, and report generation.
"""

import os
import sys
import json
import datetime
import logging
import importlib
import glob
from typing import Dict, List, Tuple, Optional, Any, Union
import jax

# Ensure the stoix module is in the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from tests.performance_tests.framework.utils import (
    test_algorithm_performance,
    TestResult,
)
from tests.performance_tests.framework.registry import get_registry

# Configure logging
logger = logging.getLogger(__name__)


def discover_tests():
    """Find and import all test modules to register their tests."""
    logger.debug("Discovering tests...")

    # Find all test_*.py files in the algorithms directory and subdirectories
    algorithms_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "algorithms"
    )

    # Get all Python files in the algorithms directory and subdirectories
    test_files = []
    for root, _, files in os.walk(algorithms_dir):
        for file in files:
            if file.endswith(".py") and not file == "__init__.py":
                test_files.append(os.path.join(root, file))

    logger.debug(f"Found {len(test_files)} test files: {test_files}")

    # Ensure the test directory is in the path
    if algorithms_dir not in sys.path:
        sys.path.insert(0, algorithms_dir)

    # Import each test file
    for test_file in test_files:
        # Fix for proper module path construction
        # Get the path relative to tests directory (2 levels up from framework)
        tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        rel_path = os.path.relpath(test_file, tests_dir)
        
        # Convert to module path (remove .py extension)
        module_path = rel_path.replace(os.path.sep, ".")[:-3]
        
        # Build the full module path
        full_module_path = f"tests.{module_path}"

        try:
            # Import the module to register its tests
            logger.debug(f"Attempting to import test module: {full_module_path}")
            module = importlib.import_module(full_module_path)
            logger.debug(f"Successfully imported module: {full_module_path}")
        except ImportError as e:
            logger.error(f"Error importing test module {full_module_path}: {str(e)}")

    # Get the registry from the test_registry module
    registry = get_registry()
    logger.debug(f"Test discovery complete. Registry has {len(registry)} entries")
    for key in registry:
        logger.debug(f"  Registered test: {key}")

    return registry


def list_available_tests() -> List[Tuple[str, str]]:
    """
    Get a list of available algorithm-environment pairs for testing.

    Returns:
        List of (algorithm, environment) tuples that have registered test functions
    """
    registry = get_registry()
    logger.debug(f"list_available_tests: Registry has {len(registry)} entries")
    return list(registry.keys())


def run_tests(
    algorithms: Optional[List[str]] = None,
    environments: Optional[List[str]] = None,
    establish_baseline: bool = False,
    max_steps: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run performance tests based on the provided filters.

    Args:
        algorithms: List of algorithms to test, or None for all
        environments: List of environments to test, or None for all
        establish_baseline: Whether to establish new baselines
        max_steps: Maximum number of training steps
        config_overrides: Overrides for the algorithm configurations

    Returns:
        Dictionary mapping test names to test results
    """
    # Get registry of available tests
    registry = get_registry()
    results = {}

    # Filter tests based on algorithm and environment
    tests_to_run = []
    for (algo, env), test_fn in registry.items():
        if (algorithms is None or algo in algorithms) and (
            environments is None or env in environments
        ):
            tests_to_run.append(((algo, env), test_fn))

    if not tests_to_run:
        logger.warning("No tests found matching the specified criteria")
        if algorithms:
            logger.warning(f"Requested algorithms: {algorithms}")
        if environments:
            logger.warning(f"Requested environments: {environments}")

        available_tests = list_available_tests()
        logger.warning(f"Available tests: {available_tests}")
        return results

    logger.info(f"Running {len(tests_to_run)} tests")

    for (algo, env), test_fn in tests_to_run:
        test_name = f"{algo}_{env}"
        logger.info(f"Running test: {test_name}")

        try:
            # Run the test
            result = test_fn(
                establish_baseline=establish_baseline,
                max_steps=max_steps,
                config_overrides=config_overrides,
            )
            results[test_name] = result
            logger.info(f"Test {test_name} completed: {result.get('success', False)}")
        except Exception as e:
            # Capture and log the full traceback for easier debugging
            error_msg = f"Error in test {test_name}: {str(e)}"
            logger.error(error_msg)
            results[test_name] = {
                "algorithm": algo,
                "environment": env,
                "success": False,
                "message": str(e),
                "metrics": {},
            }

    return results


def generate_report(results: Dict[str, Dict[str, Any]], output_dir: str = None) -> str:
    """Generate a performance test report.

    Args:
        results: Dictionary mapping test names to test results
        output_dir: Directory to save the report to

    Returns:
        Path to the saved report file
    """
    report_lines = [
        "# Performance Test Report",
        f"Generated: {datetime.datetime.now()}",
    ]

    # Skip report generation if no results
    if not results:
        report_lines.append("\nNo tests were run.")
        return "\n".join(report_lines)

    # Summary section
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    baseline_tests = sum(
        1 for r in results.values() if r.get("established_baseline", False)
    )

    report_lines.extend(
        [
            "",
            "## Summary",
            f"- Total tests: {total_tests}",
            f"- Successful tests: {successful_tests}",
            f"- Failed tests: {total_tests - successful_tests}",
            f"- Tests establishing baselines: {baseline_tests}",
            "",
        ]
    )

    # Results table
    report_lines.extend(
        [
            "## Results",
            "",
            "| Algorithm | Environment | Status | Performance vs Baseline |",
            "| --------- | ----------- | ------ | ----------------------- |",
        ]
    )

    for test_name, result in results.items():
        algo = result["algorithm"]
        env = result["environment"]
        status = (
            "✅ Success"
            if result.get("success", False)
            else f"❌ Failed: {result['message']}"
        )

        if result.get("established_baseline", False):
            comparison = "Baseline Established"
        else:
            comparison = "N/A"

        report_lines.append(f"| {algo} | {env} | {status} | {comparison} |")

    # Detailed results
    report_lines.extend(["", "## Detailed Results", ""])

    for test_name, result in results.items():
        report_lines.extend(
            [
                f"### {result['algorithm']} - {result['environment']}",
                "",
                f"Status: {'Success' if result.get('success', False) else 'Failed'}",
                "",
            ]
        )

        if result.get("message"):
            report_lines.extend([f"Error: {result['message']}", ""])

        if result.get("established_baseline", False):
            report_lines.extend(["Established new baseline.", ""])

        # Metrics comparison
        if result.get("metrics", {}) and result.get("baseline_metrics", {}):
            report_lines.extend(
                [
                    "#### Metrics Comparison",
                    "",
                    "| Metric | Current | Baseline | Diff (%) | Status |",
                    "| ------ | ------- | -------- | -------- | ------ |",
                ]
            )

            for metric_name in sorted(result["metrics"].keys()):
                if metric_name in result.get("baseline_metrics", {}):
                    current = result["metrics"][metric_name]
                    baseline = result["baseline_metrics"][metric_name]
                    pct_diff = (
                        ((current - baseline) / abs(baseline)) * 100
                        if baseline != 0
                        else float("inf")
                    )

                    # Determine if the change is an improvement or regression
                    # This assumes higher values are better, which is true for rewards
                    # For metrics where lower is better, this logic would need to be inverted
                    status = "🟢" if pct_diff > 0 else "🔴" if pct_diff < 0 else "⚪"

                    report_lines.append(
                        f"| {metric_name} | {current:.4f} | {baseline:.4f} | {pct_diff:+.2f}% | {status} |"
                    )

            report_lines.append("")

    report = "\n".join(report_lines)

    # Save report if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"performance_report_{timestamp}.md")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        return report_path

    return report
