#!/usr/bin/env python3
"""
Performance Test Runner for Stoix Reinforcement Learning Algorithms

This module provides the core functionality for running performance tests,
including test discovery, execution, and report generation.
"""

import importlib
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from uuid import uuid4

from stoix.tests.performance_tests.framework.registry import get_registry
from stoix.tests.performance_tests.framework.utils import (
    MAIN_PERFORMANCE_METRICS,
    TestResult,
)

# Configure logging
logger = logging.getLogger(__name__)


def discover_tests() -> Dict[Tuple[str, str, str, str], Callable[..., Any]]:
    """
    Discover and import all test modules to register tests.

    This function dynamically discovers and imports all test modules
    in the tests/performance_tests/algorithms directory to ensure they
    are registered through the @register_test decorator.

    Returns:
        Dictionary mapping (algorithm, environment, module_path, arch) tuples to test functions
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


def list_available_tests() -> List[Tuple[str, str, str, str]]:
    """
    Get a list of available algorithm-environment pairs for testing.

    Returns:
        List of (algorithm, environment, module_path, arch) tuples that have registered test functions
    """
    registry = get_registry()
    logger.debug(f"Available tests: {len(registry)}")
    return list(registry.keys())


def run_tests(
    algorithms: Optional[List[str]] = None,
    environments: Optional[List[str]] = None,
    env_suites: Optional[List[str]] = None,
    establish_baseline: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
    num_seeds: int = 1,
    use_slurm: bool = False,
    slurm_config: str = "slurm",
) -> Dict[str, "TestResult"]:
    """
    Run performance tests for specified algorithms and environments.

    Args:
        algorithms: List of algorithms to test (if None, test all)
        environments: List of environments to test (if None, test all)
        env_suites: List of environment suites to test (if None, test all)
        establish_baseline: Whether to establish new baselines
        config_overrides: Configuration overrides to apply to all tests
        num_seeds: Number of seeds to run for each test
        use_slurm: Whether to use SLURM for parallel execution
        slurm_config: Name of the Slurm config file to use (without .yaml extension)

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

        # Filter by specific environment
        if environments and env not in environments:
            # Check if this environment matches an environment suite
            if env_suites:
                # Extract suite part (first part before '/') from environment
                env_suite = env.split("/")[0] if "/" in env else env
                if env_suite not in env_suites:
                    continue
            else:
                continue
        # Filter by environment suite if no specific environments provided
        elif env_suites and not environments:
            # Extract suite part (first part before '/') from environment
            env_suite = env.split("/")[0] if "/" in env else env
            if env_suite not in env_suites:
                continue

        tests_to_run[(algo, env, module_path, arch)] = test_func

    if not tests_to_run:
        if algorithms or environments or env_suites:
            logger.warning(
                f"No tests found for specified algorithms({algorithms}), environments({environments}), or environment suites({env_suites})"
            )
        else:
            logger.warning("No tests found")
        return {}

    logger.info(f"Running {len(tests_to_run)} tests with {num_seeds} seeds each")

    # If using SLURM, use submitit to run tests in parallel
    if use_slurm:
        return run_tests_slurm(
            tests_to_run, establish_baseline, config_overrides, num_seeds, slurm_config
        )

    # Run the tests sequentially
    results = {}
    for (algo, env, module_path, arch), test_func in tests_to_run.items():
        test_name = f"{algo}_{env.replace('/', '_')}_{arch}"
        logger.info(f"Running test: {test_name}")

        try:
            # Run the test function with the appropriate arguments
            if config_overrides is None:
                config_overrides = {}
            config_overrides.update({"logger.use_json": "True"})
            config_overrides.update(
                {"logger.base_exp_path": f"stoix/tests/performance_tests/data/experiment_runs"}
            )
            config_overrides.update(
                {
                    "logger.kwargs.json_path": f"{test_name}/{datetime.now().strftime('%Y%m%d%H%M%S') + str(uuid4())}"
                }
            )
            result = test_func(
                establish_baseline=establish_baseline,
                config_overrides=config_overrides,
                num_seeds=num_seeds,
            )

            # Add result to results dictionary
            results[test_name] = result

            logger.info(f"Test {test_name} completed: {result.summary}")

        except Exception as e:
            logger.error(f"Error running test {test_name}: {str(e)}")
            logger.error(traceback.format_exc())

            # Record error as a failed test
            results[test_name] = TestResult(
                algorithm=algo,
                environment=env,
                success=False,
                message=f"Exception occurred: {str(e)}",
            )

    # Return test results
    return results


def generate_report(results: Dict[str, "TestResult"], output_dir: Optional[str] = None) -> str:
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

    # Add results table
    report_lines.extend(
        [
            "## Results",
            "",
            "| Algorithm | Environment | Status | Notes |",
            "| --------- | ----------- | ------ | ----- |",
        ]
    )

    # Generate results table
    for test_name, result in results.items():
        algo = result.algorithm
        env = result.environment
        status = "✅ Success" if result.success else "❌ Failed"

        notes = []
        if result.established_baseline:
            notes.append("Baseline established")
        if result.num_seeds > 1:
            notes.append(f"Ran with {result.num_seeds} seeds")
        if not result.success and result.message:
            notes.append(result.message)

        notes_str = ", ".join(notes) if notes else ""
        report_lines.append(f"| {algo} | {env} | {status} | {notes_str} |")

    # Add detailed metrics section
    has_metrics = any(bool(r.metrics) for r in results.values())

    if has_metrics:
        report_lines.extend(["", "## Detailed Metrics", ""])

        for test_name, result in results.items():
            if not result.metrics:
                continue

            report_lines.extend(
                [
                    f"### {result.algorithm} - {result.environment}",
                    "",
                ]
            )

            # If we have baseline metrics, show comparison
            if result.baseline_metrics:
                # Check if we have multi-seed statistics
                if result.num_seeds > 1 and result.metric_stats:
                    report_lines.extend(
                        [
                            "| Metric | Current (Mean ± 95% CI) | Baseline (Mean ± 95% CI) | Change |",
                            "| ------ | ----------------------- | ------------------------ | ------ |",
                        ]
                    )

                    for metric, value in sorted(result.metrics.items()):
                        if metric in result.baseline_metrics:
                            baseline = result.baseline_metrics[metric]
                            change = value - baseline
                            change_str = f"{change:+.4f}"

                            # Add confidence intervals if available
                            if metric in result.metric_stats:
                                stats = result.metric_stats[metric]
                                ci95 = stats.get("ci95", 0)
                                value_str = f"{value:.4f} ± {ci95:.4f}"
                            else:
                                value_str = f"{value:.4f}"

                            # Add baseline statistics if available
                            if metric in result.baseline_statistics:
                                baseline_stats = result.baseline_statistics[metric]
                                baseline_ci95 = baseline_stats.get("ci95", 0)
                                baseline_str = f"{baseline:.4f} ± {baseline_ci95:.4f}"
                            else:
                                baseline_str = f"{baseline:.4f}"

                            report_lines.append(
                                f"| {metric} | {value_str} | {baseline_str} | {change_str} |"
                            )
                else:
                    report_lines.extend(
                        [
                            "| Metric | Current | Baseline | Change |",
                            "| ------ | ------- | -------- | ------ |",
                        ]
                    )

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
                if result.num_seeds > 1 and result.metric_stats:
                    report_lines.extend(
                        [
                            "| Metric | Mean | Std Dev | Min | Max | 95% CI |",
                            "| ------ | ---- | ------- | --- | --- | ------ |",
                        ]
                    )

                    for metric, value in sorted(result.metrics.items()):
                        if metric in result.metric_stats:
                            stats = result.metric_stats[metric]
                            report_lines.append(
                                f"| {metric} | {value:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | ±{stats['ci95']:.4f} |"
                            )
                        else:
                            report_lines.append(
                                f"| {metric} | {value:.4f} | N/A | N/A | N/A | N/A |"
                            )
                else:
                    report_lines.extend(
                        [
                            "| Metric | Value |",
                            "| ------ | ----- |",
                        ]
                    )

                    for metric, value in sorted(result.metrics.items()):
                        report_lines.append(f"| {metric} | {value:.4f} |")

            # Add seed details if multiple seeds were used
            if result.num_seeds > 1 and result.seed_metrics:
                report_lines.extend(
                    [
                        "",
                        "#### Individual Seed Results",
                        "",
                    ]
                )

                # Get all unique metrics across all seeds
                all_metrics = set()
                for seed_data in result.seed_metrics.values():
                    all_metrics.update(seed_data.keys())

                # Get primary metrics first (if they exist)
                primary_metrics = [m for m in MAIN_PERFORMANCE_METRICS if m in all_metrics]
                other_metrics = sorted(all_metrics - set(primary_metrics))
                display_metrics = (
                    primary_metrics + other_metrics[:3]
                )  # Show primary metrics + up to 3 others

                # Create header
                header = "| Seed | " + " | ".join(display_metrics) + " |"
                separator = "| ---- | " + " | ".join(["-" * len(m) for m in display_metrics]) + " |"

                report_lines.extend([header, separator])

                # Add rows for each seed
                for seed, seed_metrics in sorted(result.seed_metrics.items()):
                    values = []
                    for metric in display_metrics:
                        value = seed_metrics.get(metric, "N/A")
                        if isinstance(value, (int, float)):
                            value = f"{value:.4f}"
                        values.append(value)

                    report_lines.append(f"| {seed} | " + " | ".join(values) + " |")

                # If we truncated metrics, note this
                if len(all_metrics) > len(display_metrics):
                    report_lines.append(
                        f"*Note: {len(all_metrics) - len(display_metrics)} additional metrics not shown*"
                    )

            # Show baseline seed information if available
            if result.baseline_seeds and any(result.baseline_seeds.values()):
                report_lines.extend(
                    [
                        "",
                        "#### Baseline Seed Data",
                        "",
                        "The baseline was established using multiple seeds. Here are the individual seed results:",
                        "",
                    ]
                )

                # Find common metrics across seeds
                baseline_metrics = set()
                for seed_data in result.baseline_seeds.values():
                    baseline_metrics.update(seed_data.keys())

                # Prioritize main metrics
                display_metrics = [m for m in MAIN_PERFORMANCE_METRICS if m in baseline_metrics]
                if not display_metrics:
                    display_metrics = sorted(baseline_metrics)[:3]

                # Create header
                header = "| Seed | " + " | ".join(display_metrics) + " |"
                separator = "| ---- | " + " | ".join(["-" * len(m) for m in display_metrics]) + " |"

                report_lines.extend([header, separator])

                # Add rows for each seed
                for seed, seed_metrics in sorted(
                    result.baseline_seeds.items(),
                    key=lambda x: int(x[0]) if x[0].isdigit() else x[0],
                ):
                    values = []
                    for metric in display_metrics:
                        value = seed_metrics.get(metric, "N/A")
                        if isinstance(value, (int, float)):
                            value = f"{value:.4f}"
                        values.append(value)

                    report_lines.append(f"| {seed} | " + " | ".join(values) + " |")

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


def run_tests_slurm(
    tests_to_run: Dict[Tuple[str, str, str, str], Any],
    establish_baseline: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None,
    num_seeds: int = 1,
    slurm_config: str = "slurm",
) -> Dict[str, TestResult]:
    """
    Run performance tests in parallel using SLURM.

    This function submits each test+seed combination as a separate SLURM job and waits for all jobs
    to complete before returning the combined results.

    Args:
        tests_to_run: Dictionary mapping (algo, env, module_path, arch) to test functions
        establish_baseline: Whether to establish new baselines
        config_overrides: Configuration overrides to apply to all tests
        num_seeds: Number of seeds to run for each test
        slurm_config: Name of the Slurm config file to use (without .yaml extension)

    Returns:
        Dictionary mapping test names to test results
    """
    import hydra
    import submitit
    from omegaconf import DictConfig, OmegaConf

    logger.info(f"Running {len(tests_to_run)} tests with {num_seeds} seeds each in parallel using SLURM")
    total_jobs = len(tests_to_run) * num_seeds if num_seeds > 1 else len(tests_to_run)
    logger.info(f"Will submit a total of {total_jobs} SLURM jobs")

    # Load Slurm configuration
    slurm_cfg_path = os.path.join("stoix", "configs", "launcher", f"{slurm_config}.yaml")
    if not os.path.exists(slurm_cfg_path):
        # Try to find the config using hydra
        try:
            with hydra.initialize_config_module(config_module="stoix.configs.launcher"):
                slurm_cfg = hydra.compose(config_name=slurm_config)
        except Exception as e:
            logger.error(f"Failed to load SLURM config '{slurm_config}': {e}")
            return {}
    else:
        # Load config directly
        slurm_cfg = OmegaConf.load(slurm_cfg_path)

    # Configure the SLURM executor
    slurm_log_folder = os.path.join("stoix", "tests", "performance_tests", "data", "slurm_logs")
    os.makedirs(slurm_log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=slurm_log_folder)
    executor.update_parameters(
        name="stoix_tests",
        nodes=1,
        gpus_per_node=slurm_cfg.slurm.gpus_per_node,
        cpus_per_task=slurm_cfg.slurm.cpus_per_task,
        time=slurm_cfg.slurm.time,
        slurm_partition=slurm_cfg.slurm.partition,
        slurm_qos=slurm_cfg.slurm.qos if hasattr(slurm_cfg.slurm, "qos") else None,
        slurm_account=slurm_cfg.slurm.account if hasattr(slurm_cfg.slurm, "account") else None,
    )

    # Dictionary to store jobs by test name
    all_jobs = {}
    start_seed = 42  # Base seed value

    # Create a wrapper function to run a single seed test
    def run_single_seed_test(
        test_key: Tuple[str, str, str, str],
        test_func: Callable[..., Any],
        seed: int,
        establish_baseline: bool,
        config_overrides: Optional[Dict[str, Any]],
    ) -> Tuple[str, int, Dict[str, float]]:
        """Run a test with a single seed and return the metrics."""
        algo, env, module_path, arch = test_key
        test_name = f"{algo}_{env.replace('/', '_')}_{arch}"
        
        try:
            # Initialize config overrides if necessary
            if config_overrides is None:
                config_overrides = {}
                
            # Add standard logging configuration
            config_overrides.update({"logger.use_json": "True"})
            config_overrides.update({"logger.base_exp_path": f"stoix/tests/performance_tests/data/experiment_runs"})
            job_id = f"{test_name}_seed{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}{str(uuid4())[:6]}"
            config_overrides.update({"logger.kwargs.json_path": job_id})

            # Import the test module's utility function to run a single seed
            from stoix.tests.performance_tests.framework.utils import run_algorithm_with_config, create_hydra_config
            
            logger.info(f"Running test {test_name} with seed {seed}")
            
            # Prepare overrides
            overrides = []
            if config_overrides:
                for key, value in config_overrides.items():
                    overrides.append(f"{key}={value}")
                    
            # Add the environment to the overrides
            overrides.append(f"env={env}")
            
            try:
                # Create the config
                config_name = f"default_{algo}"
                cfg = create_hydra_config(
                    config_path=f"../../../configs/default/{arch}",
                    config_name=config_name,
                    overrides=overrides,
                )
                
                # Run the algorithm with the seed
                metrics = run_algorithm_with_config(cfg, module_path, mock=False, seed=seed)
                return test_name, seed, metrics
                
            except Exception as e:
                logger.error(f"Error running {test_name} with seed {seed}: {str(e)}")
                raise
                
        except Exception as e:
            # Return empty metrics on error
            return test_name, seed, {"error": 1.0, "error_message": str(e)}

    # Submit all jobs
    with executor.batch():
        for test_key, test_func in tests_to_run.items():
            algo, env, module_path, arch = test_key
            test_name = f"{algo}_{env.replace('/', '_')}_{arch}"
            
            # Initialize job list for this test
            all_jobs[test_name] = {
                "algo": algo,
                "env": env,
                "jobs": []
            }
            
            # If we have multiple seeds, run each as a separate job
            if num_seeds > 1:
                for seed_idx in range(num_seeds):
                    current_seed = start_seed + seed_idx
                    logger.info(f"Submitting SLURM job for test: {test_name} with seed {current_seed}")
                    
                    job = executor.submit(
                        run_single_seed_test,
                        test_key,
                        test_func,
                        current_seed,
                        establish_baseline,
                        config_overrides,
                    )
                    all_jobs[test_name]["jobs"].append(job)
            else:
                # Single seed case - just run with default seed
                logger.info(f"Submitting SLURM job for test: {test_name}")
                job = executor.submit(
                    run_single_seed_test,
                    test_key,
                    test_func,
                    start_seed,
                    establish_baseline,
                    config_overrides,
                )
                all_jobs[test_name]["jobs"].append(job)

    # Wait for all jobs to complete
    logger.info(f"Waiting for {total_jobs} SLURM jobs to complete...")
    
    # Simple progress tracking
    completed = 0
    while completed < total_jobs:
        new_completed = sum(sum(1 for job in test_info["jobs"] if job.done()) 
                           for test_info in all_jobs.values())
        if new_completed > completed:
            completed = new_completed
            logger.info(f"Progress: {completed}/{total_jobs} jobs completed ({completed/total_jobs:.1%})")
        time.sleep(5)
    
    # Process results and create TestResult objects
    results = {}
    
    for test_name, test_info in all_jobs.items():
        # Create a TestResult object for this test
        result = TestResult(
            algorithm=test_info["algo"],
            environment=test_info["env"],
            success=True,  # Will be updated if any failures occur
            num_seeds=num_seeds,
        )
        
        try:
            # Collect results from all seed jobs
            for job in test_info["jobs"]:
                try:
                    job_test_name, seed, metrics = job.result()
                    
                    # Add the seed metrics to the result
                    result.add_seed_run(seed, metrics)
                    
                    # Check for errors in the metrics
                    if "error" in metrics:
                        result.success = False
                        result.message = f"Error in seed {seed}: {metrics.get('error_message', 'Unknown error')}"
                        logger.error(f"Error in {test_name} seed {seed}: {result.message}")
                        
                except Exception as e:
                    # Record failure for this seed
                    result.success = False
                    result.message = f"Job failed: {str(e)}"
                    logger.error(f"Failed to get result for job in test {test_name}: {str(e)}")
            
            # Process baseline if needed
            if use_baseline := not establish_baseline:
                # Import baseline functions
                from stoix.tests.performance_tests.framework.utils import (
                    load_baseline, 
                    compare_metrics,
                    save_baseline
                )
                
                baseline_data = load_baseline(test_info["algo"], test_info["env"])
                
                # Extract baseline information
                if baseline_data:
                    # Extract baseline metrics
                    if "metrics" in baseline_data:
                        result.baseline_metrics = baseline_data["metrics"]
                        
                        # Store baseline statistics if available
                        if "statistics" in baseline_data:
                            result.baseline_statistics = baseline_data["statistics"]
                            
                        # Store baseline seed data if available
                        if "seeds" in baseline_data:
                            result.baseline_seeds = baseline_data["seeds"]
                            
                        # Compare to baseline
                        result.comparison = compare_metrics(result.metrics, result.baseline_metrics)
                
            # Establish new baseline if requested
            if establish_baseline:
                from stoix.tests.performance_tests.framework.utils import save_baseline
                
                if save_baseline(
                    test_info["algo"], 
                    test_info["env"], 
                    result.metrics, 
                    result.metric_stats, 
                    result.seed_metrics
                ):
                    result.established_baseline = True
                    result.message = f"Established new baseline for {test_info['algo']} on {test_info['env']}"
                    logger.info(result.message)
                else:
                    result.success = False
                    result.message = f"Failed to establish baseline for {test_info['algo']} on {test_info['env']}"
                    logger.error(result.message)
            
            # Add to results
            results[test_name] = result
            logger.info(f"Test {test_name} completed: {result.summary}")
            
        except Exception as e:
            # Create a failed result
            results[test_name] = TestResult(
                algorithm=test_info["algo"],
                environment=test_info["env"],
                success=False,
                message=f"Error processing results: {str(e)}",
            )
            logger.error(f"Error processing results for {test_name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.info(f"All {total_jobs} SLURM jobs completed and processed")
    return results
