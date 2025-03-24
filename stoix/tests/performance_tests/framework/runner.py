#!/usr/bin/env python3
"""
Performance Test Runner for Stoix Reinforcement Learning Algorithms

This module provides the core functionality for running performance tests,
including test discovery, execution, and report generation.
"""

import os
from datetime import datetime
import logging
import importlib
import traceback
from typing import Dict, List, Tuple, Optional, Any
from uuid import uuid4

from stoix.tests.performance_tests.framework.registry import get_registry


# Configure logging
logger = logging.getLogger(__name__)


def discover_tests() -> Dict[Tuple[str, str], Any]:
    """
    Discover and import all test modules to register tests.
    
    This function dynamically discovers and imports all test modules
    in the tests/performance_tests/algorithms directory to ensure they
    are registered through the @register_test decorator.
    
    Returns:
        Dictionary mapping (algorithm, environment) tuples to test functions
    """
    from stoix.tests.performance_tests.framework.registry import get_registry
    
    logger.info("Discovering tests...")
    
    # Get the base package path
    base_path = os.path.join(os.path.dirname(__file__), "..", "algorithms")
    
    # Get all algorithm directories
    if not os.path.exists(base_path):
        logger.warning(f"Algorithms directory not found at {base_path}")
        return {}
    
    # Import test modules
    count = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                # Convert file path to module path
                rel_path = os.path.relpath(os.path.join(root, file), os.path.join(base_path, ".."))
                module_path = rel_path.replace(os.path.sep, ".").replace(".py", "")
                
                # Import the module
                try:
                    logger.debug(f"Importing {module_path}")
                    importlib.import_module(f"stoix.tests.performance_tests.{module_path}")
                    count += 1
                except ImportError as e:
                    logger.error(f"Failed to import {module_path}: {str(e)}")
    
    # Get the registry after importing all modules
    registry = get_registry()
    logger.info(f"Discovered {count} test modules with {len(registry)} tests")
    
    return registry


def list_available_tests() -> List[Tuple[str, str]]:
    """
    Get a list of available algorithm-environment pairs for testing.
    
    Returns:
        List of (algorithm, environment) tuples that have registered test functions
    """
    registry = get_registry()
    logger.debug(f"Available tests: {len(registry)}")
    return list(registry.keys())


def run_tests(
    algorithms: Optional[List[str]] = None,
    environments: Optional[List[str]] = None,
    establish_baseline: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run performance tests for specified algorithms and environments.
    
    Args:
        algorithms: List of algorithms to test (if None, test all)
        environments: List of environments to test (if None, test all)
        establish_baseline: Whether to establish new baselines
        config_overrides: Configuration overrides to apply to all tests
        
    Returns:
        Dictionary mapping test names to test results
    """
    
    # Get the registry of tests
    registry = get_registry()
    
    if not registry:
        logger.warning("No tests are registered")
        return {}
    
    # Filter tests by algorithm and environment if specified
    tests_to_run = {}
    for (algo, env, module_path, arch), test_func in registry.items():
        if algorithms and algo not in algorithms:
            continue
        if environments and env not in environments:
            continue
        tests_to_run[(algo, env, module_path, arch)] = test_func
    
    if not tests_to_run:
        if algorithms or environments:
            logger.warning(f"No tests found for specified algorithms({algorithms}) and environments({environments})")
        else:
            logger.warning("No tests found")
        return {}
    
    logger.info(f"Running {len(tests_to_run)} tests")
    
    # Run the tests
    results = {}
    for (algo, env, module_path, arch), test_func in tests_to_run.items():
        test_name = f"{algo}_{env.replace('/', '_')}_{arch}"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Run the test function with the appropriate arguments
            config_overrides.update({"logger.use_json" : "True"})
            config_overrides.update({"logger.base_exp_path" : f"stoix/tests/performance_tests/data/experiment_runs"})
            config_overrides.update({"logger.kwargs.json_path" : f"{test_name}/{datetime.now().strftime('%Y%m%d%H%M%S') + str(uuid4())}"})
            result = test_func(
                establish_baseline=establish_baseline,
                config_overrides=config_overrides,
            )
            
            # Add result to results dictionary
            results[test_name] = result
            
            logger.info(f"Test {test_name} completed: {result.summary}")
            
        except Exception as e:
            logger.error(f"Error running test {test_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Record error as a failed test
            from stoix.tests.performance_tests.framework.utils import TestResult
            results[test_name] = TestResult(
                algorithm=algo,
                environment=env,
                success=False,
                message=f"Exception occurred: {str(e)}",
            )
    
    # Return test results
    return results


def generate_report(
    results: Dict[str, Dict[str, Any]], 
    output_dir: Optional[str] = None
) -> str:
    """
    Generate a performance test report.
    
    Args:
        results: Dictionary mapping test names to test results
        output_dir: Directory to save the report to
    
    Returns:
        Report content as string, or path to the saved report file
    """
    timestamp = datetime.now()
    report_lines = [
        "# Performance Test Report",
        f"Generated: {timestamp}",
    ]
    
    if not results:
        report_lines.append("\nNo tests were run.")
        return "\n".join(report_lines)
    
    # Compute summary statistics
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r.success)
    baseline_tests = sum(1 for r in results.values() if r.established_baseline)
    
    # Add summary section
    report_lines.extend([
        "",
        "## Summary",
        f"- Total tests: {total_tests}",
        f"- Successful tests: {successful_tests}",
        f"- Failed tests: {total_tests - successful_tests}",
        f"- Tests establishing baselines: {baseline_tests}",
        "",
    ])
    
    # Add results table
    report_lines.extend([
        "## Results",
        "",
        "| Algorithm | Environment | Status | Notes |",
        "| --------- | ----------- | ------ | ----- |",
    ])
    
    # Generate results table
    for test_name, result in results.items():
        algo = result.algorithm
        env = result.environment
        status = "✅ Success" if result.success else "❌ Failed"
        
        notes = []
        if result.established_baseline:
            notes.append("Baseline established")
        if not result.success and result.message:
            notes.append(result.message)
            
        notes_str = ", ".join(notes) if notes else ""
        report_lines.append(f"| {algo} | {env} | {status} | {notes_str} |")
    
    # Add detailed metrics section
    has_metrics = any(
        bool(r.metrics) for r in results.values()
    )
    
    if has_metrics:
        report_lines.extend(["", "## Detailed Metrics", ""])
        
        for test_name, result in results.items():
            if not result.metrics:
                continue
                
            report_lines.extend([
                f"### {result.algorithm} - {result.environment}",
                "",
            ])
            
            # If we have baseline metrics, show comparison
            if result.baseline_metrics:
                report_lines.extend([
                    "| Metric | Current | Baseline | Change |",
                    "| ------ | ------- | -------- | ------ |",
                ])
                
                for metric, value in sorted(result.metrics.items()):
                    if metric in result.baseline_metrics:
                        baseline = result.baseline_metrics[metric]
                        change = value - baseline
                        change_str = f"{change:+.4f}"
                        report_lines.append(
                            f"| {metric} | {value:.4f} | {baseline:.4f} | {change_str} |"
                        )
            else:
                # Just show current metrics
                report_lines.extend([
                    "| Metric | Value |",
                    "| ------ | ----- |",
                ])
                
                for metric, value in sorted(result.metrics.items()):
                    report_lines.append(f"| {metric} | {value:.4f} |")
            
            report_lines.append("")
    
    report = "\n".join(report_lines)
    
    # Save report if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"performance_report_{timestamp_str}.md")
        
        try:
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to save report to {report_path}: {str(e)}")
    
    return report
