#!/usr/bin/env python3
"""
Performance Test Utilities for Stoix Reinforcement Learning Algorithms

This module provides utility functions for running performance tests on
reinforcement learning algorithms. It handles:

1. Loading and processing Hydra configurations
2. Running algorithm experiments with specified configurations
3. Managing baselines (loading, saving, comparing)
4. Processing and comparing performance metrics

The main entry point is the `test_algorithm_performance` function, which handles
all aspects of running a test and comparing results to baselines.
"""

import importlib
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

# Constants
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "baselines")

# Primary metrics used for determining performance changes (in priority order)
MAIN_PERFORMANCE_METRICS = [
    "eval_return",  # Average evaluation return
    "eval_episode_return",  # Average evaluation episode return
    "return",  # Average training return
    "episode_return",  # Average training episode return
    "success_rate",  # Task success rate (if applicable)
    "reward",  # Raw reward value
]

# Confidence level for intervals (0.95 = 95% confidence)
CONFIDENCE_LEVEL = 0.95


@dataclass
class TestResult:
    """
    Class to store the results of a performance test.

    This dataclass encapsulates all information about a test run,
    including performance metrics, baseline comparisons, and error states.
    """

    algorithm: str
    environment: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    comparison: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    established_baseline: bool = False

    # Multi-seed run data
    seed_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    metric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_seeds: int = 1

    # Run identifier for tracking experiment data
    run_id: Optional[str] = None

    # Extended baseline information
    baseline_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_seeds: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Get a concise summary of the test result for logging."""
        if not self.success:
            return f"Failed: {self.message}"

        if self.established_baseline:
            return "Baseline established"

        main_metric = self._find_main_performance_metric()
        if main_metric:
            metric_name, value, baseline, pct_diff = main_metric
            if self.num_seeds > 1 and metric_name in self.metric_stats:
                stats = self.metric_stats[metric_name]
                return f"{metric_name}: {value:.2f} ± {stats['ci95']:.2f} vs {baseline:.2f} ({pct_diff:+.2f}%)"
            else:
                return f"{metric_name}: {value:.2f} vs {baseline:.2f} ({pct_diff:+.2f}%)"

        return "Success (no metrics available)"

    def _find_main_performance_metric(self) -> Optional[Tuple[str, float, float, float]]:
        """Find the most important performance metric that exists in both current and baseline."""
        if not self.metrics or not self.baseline_metrics:
            return None

        # First check the priority list
        for metric_name in MAIN_PERFORMANCE_METRICS:
            if metric_name in self.metrics and metric_name in self.baseline_metrics:
                return self._create_metric_comparison(metric_name)

        # If no priority metric found, use the first common metric alphabetically
        common_metrics = sorted(set(self.metrics) & set(self.baseline_metrics))
        if common_metrics:
            return self._create_metric_comparison(common_metrics[0])

        return None

    def _create_metric_comparison(self, metric_name: str) -> Tuple[str, float, float, float]:
        """Create a comparison tuple for the given metric."""
        value = self.metrics[metric_name]
        baseline = self.baseline_metrics[metric_name]

        # Calculate percentage difference safely
        if baseline != 0:
            pct_diff = ((value - baseline) / abs(baseline)) * 100
        else:
            pct_diff = float("inf") if value > 0 else float("-inf") if value < 0 else 0.0

        return (metric_name, value, baseline, pct_diff)

    def add_seed_run(self, seed: int, metrics: Dict[str, float]) -> None:
        """
        Add results from a single seed run.

        Args:
            seed: The random seed used for this run
            metrics: Performance metrics from this run
        """
        self.seed_metrics[seed] = metrics
        self.num_seeds = len(self.seed_metrics)

        # Update the main metrics dictionary with aggregated values
        self._calculate_stats()

    def _calculate_stats(self) -> None:
        """Calculate statistics across all seed runs."""
        if not self.seed_metrics:
            return

        # First, identify all unique metrics across all seeds
        all_metrics: set[str] = set()
        for seed_data in self.seed_metrics.values():
            all_metrics.update(seed_data.keys())

        # For each metric, calculate statistics
        self.metric_stats = {}
        self.metrics = {}

        for metric in all_metrics:
            # Collect all values for this metric across all seeds
            values = []
            for seed_data in self.seed_metrics.values():
                if metric in seed_data:
                    values.append(seed_data[metric])

            if not values:
                continue

            # Convert to numpy array for statistical operations
            values_array = np.array(values)

            # Calculate mean (this will be our primary metric value)
            mean = float(np.mean(values_array))
            self.metrics[metric] = mean

            # Calculate additional statistics if we have multiple seeds
            if len(values) > 1:
                std = float(np.std(values_array, ddof=1))  # Sample standard deviation
                sem = float(stats.sem(values_array))  # Standard error of the mean

                # Calculate 95% confidence interval
                # For small sample sizes, use t-distribution
                n = len(values)
                ci95 = float(stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1) * sem)

                # Store the statistics
                self.metric_stats[metric] = {
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95": ci95,
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "count": n,
                }


# ---- Baseline Management Functions ----


def get_baseline_path(algorithm: str, environment: str) -> str:
    """
    Get the path to the baseline file for a given algorithm and environment.

    Args:
        algorithm: Name of the algorithm
        environment: Name of the environment

    Returns:
        Path to the baseline file
    """
    # Replace slashes with underscores to create a valid filename
    safe_env_name = environment.replace("/", "_").replace("\\", "_")
    filename = f"{algorithm}_{safe_env_name}.json"
    return os.path.join(BASELINE_DIR, filename)


def load_baseline(algorithm: str, environment: str) -> Dict[str, Any]:
    """
    Load baseline metrics for a given algorithm and environment.

    Returns:
        Dictionary containing baseline metrics and statistical data
    """
    baseline_path = get_baseline_path(algorithm, environment)

    if not os.path.exists(baseline_path):
        logger.info(f"No baseline found at {baseline_path}")
        return {}

    try:
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

        logger.info(f"Loaded baseline from {baseline_path}")
        return baseline_data
    except json.JSONDecodeError:
        logger.warning(f"Failed to load baseline from {baseline_path}. Invalid JSON.")
    except Exception as e:
        logger.warning(f"Error loading baseline from {baseline_path}: {str(e)}")

    return {}


def save_baseline(
    algorithm: str,
    environment: str,
    metrics: Dict[str, float],
    metric_stats: Optional[Dict[str, Dict[str, float]]] = None,
    seeds_info: Optional[Dict[int, Dict[str, float]]] = None,
) -> bool:
    """
    Save metrics as baseline for a given algorithm and environment.

    Args:
        algorithm: Name of the algorithm
        environment: Name of the environment
        metrics: Dictionary of metric means
        metric_stats: Optional dictionary of statistical metrics (std, min, max, etc.)
        seeds_info: Optional dictionary containing per-seed metrics

    Returns:
        True if saving was successful, False otherwise.
    """
    baseline_path = get_baseline_path(algorithm, environment)

    # Create baseline data structure with richer information
    baseline_data = {
        "metrics": metrics,
        "meta": {
            "created_at": datetime.now().isoformat(),
            "contains_statistics": metric_stats is not None,
            "num_seeds": len(seeds_info) if seeds_info else 1,
        },
    }

    # Add statistical metrics if available
    if metric_stats:
        baseline_data["statistics"] = metric_stats

    # Add per-seed information if available (but limit to key metrics to avoid huge files)
    if seeds_info:
        # Only store seed data for main performance metrics to keep file size reasonable
        filtered_seed_data = {}
        for seed, seed_metrics in seeds_info.items():
            filtered_metrics = {}
            for metric_name in MAIN_PERFORMANCE_METRICS:
                if metric_name in seed_metrics:
                    filtered_metrics[metric_name] = seed_metrics[metric_name]
            if filtered_metrics:
                filtered_seed_data[str(seed)] = filtered_metrics

        if filtered_seed_data:
            baseline_data["seeds"] = filtered_seed_data

    try:
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_data, f, indent=2)
        logger.info(f"Saved baseline to {baseline_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save baseline to {baseline_path}: {str(e)}")
        return False


def compare_metrics(current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    """
    Compare current metrics to baseline metrics.

    Args:
        current: Current metrics
        baseline: Baseline metrics

    Returns:
        Dictionary of differences (current - baseline) for each metric
    """
    if not current or not baseline:
        return {}

    # Find common metrics in both current and baseline
    common_metrics = set(current.keys()) & set(baseline.keys())

    # Calculate differences
    differences = {}
    for metric in common_metrics:
        differences[metric] = current[metric] - baseline[metric]

    return differences


# ---- Configuration and Experiment Functions ----


def create_hydra_config(config_path: str, config_name: str, overrides: List[str]) -> DictConfig:
    """Create a Hydra configuration with the given overrides."""
    logger.debug(f"Creating config from {config_path}/{config_name} with overrides: {overrides}")

    try:
        with hydra.initialize(version_base="1.2", config_path=config_path):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
        return cfg
    except Exception as e:
        logger.error(f"Error creating Hydra config: {str(e)}")
        raise ValueError(f"Failed to create configuration: {str(e)}")


def process_metrics(raw_metrics: Any) -> Dict[str, float]:
    """
    Process raw metrics from algorithm run into a standard format.

    Args:
        raw_metrics: Raw metrics from algorithm run, can be in various formats

    Returns:
        Processed metrics as dictionary mapping metric names to float values
    """
    metrics: Dict[str, float] = {}

    if raw_metrics is None:
        return metrics

    # Handle different types of metrics
    if isinstance(raw_metrics, dict):
        # Extract numeric metrics only
        for key, value in raw_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            # Handle nested dictionary case
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        metrics[f"{key}.{nested_key}"] = float(nested_value)
            # Handle array-like case
            elif hasattr(value, "__len__") and hasattr(value, "__getitem__"):
                try:
                    # Try to get the last value if it's a sequence
                    last_value = value[-1]
                    if isinstance(last_value, (int, float)):
                        metrics[key] = float(last_value)
                except (IndexError, TypeError):
                    pass

    # Handle numpy arrays
    elif hasattr(raw_metrics, "item") and callable(getattr(raw_metrics, "item")):
        # Convert numpy scalar to Python float
        try:
            metrics["value"] = float(raw_metrics.item())
        except (ValueError, TypeError):
            pass

    # Handle lists and simple values
    elif isinstance(raw_metrics, (list, tuple)) and raw_metrics:
        try:
            # Try to get the last value if it's a sequence
            metrics["value"] = float(raw_metrics[-1])
        except (ValueError, TypeError, IndexError):
            pass
    elif isinstance(raw_metrics, (int, float)):
        metrics["value"] = float(raw_metrics)

    return metrics


def get_run_experiment_function(module_path: str) -> Callable[[DictConfig], Any]:
    """Get the training function from the algorithm module."""
    module = importlib.import_module(module_path)
    return getattr(module, "run_experiment")


def run_algorithm_with_config(
    cfg: DictConfig, module_path: str, mock: bool = False, seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Run an algorithm with the given configuration and return performance metrics.

    Args:
        cfg: Hydra configuration to use for the algorithm
        module_path: Path to the module containing the algorithm
        mock: If True, return mock metrics instead of running the algorithm
        seed: Optional random seed to use for this run

    Returns:
        Dictionary of performance metrics
    """
    if mock:
        logger.info("Using mock metrics (test mode)")
        # Return mock metrics for testing
        import random

        # If seed is provided, set the random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        metrics = {
            "episode_return": random.uniform(800, 1200),
            "evaluation_return": random.uniform(900, 1300),
            "success_rate": random.uniform(0.7, 0.95),
            "steps_per_second": random.uniform(500, 2000),
            "total_steps": 1000000,
            "wall_time": random.uniform(1000, 3000),
        }
        return metrics

    # Normal execution path
    logger.info(f"Running algorithm with config: {cfg.system.system_name} on {cfg.env}")

    OmegaConf.set_struct(cfg, False)

    # If seed is provided, set it in the config
    if seed is not None:
        logger.info(f"Setting seed to {seed}")
        cfg.arch.seed = seed

    try:
        # Import the algorithm module and get the appropriate run_experiment function
        run_experiment = get_run_experiment_function(module_path)

        # Run the training with the provided config
        logger.debug("Starting training run")
        results = run_experiment(cfg)
        logger.debug("Training run completed")

        # Process metrics from the results
        metrics = process_metrics(results)
        logger.info(f"Collected {len(metrics)} metrics from training run")

        return metrics
    except Exception as e:
        error_msg = f"Error running algorithm: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg)


def test_algorithm_performance(
    algorithm: str,
    environment: str,
    module_path: str,
    arch: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    use_baseline: bool = True,
    establish_baseline: bool = False,
    mock: bool = False,
    num_seeds: int = 1,
    start_seed: int = 42,
) -> TestResult:
    """
    Test the performance of an algorithm on a specific environment.

    This function runs the algorithm, collects performance metrics,
    and compares them to the baseline if available.

    Args:
        algorithm: Name of the algorithm to test
        environment: Name of the environment to test on
        module_path: Path to the file containing the algorithm implementation
        arch: Architecture identifier used for configuration
        config_overrides: Optional configuration overrides
        use_baseline: Whether to compare to baseline
        establish_baseline: Whether to establish a new baseline
        mock: If True, use mock metrics instead of running the real algorithm
        num_seeds: Number of seeds (runs) to perform for statistical analysis
        start_seed: Starting seed value, will increment for each run

    Returns:
        TestResult object with test results
    """
    logger.info(f"Testing {algorithm} on {environment} with {num_seeds} seeds")

    # Initialize result object
    result = TestResult(
        algorithm=algorithm,
        environment=environment,
        success=False,
        num_seeds=num_seeds,
    )

    # Prepare config overrides
    overrides = []
    if config_overrides:
        for key, value in config_overrides.items():
            overrides.append(f"{key}={value}")

    # Add the environment to the overrides
    overrides.append(f"env={environment}")

    try:
        # For each seed, run a separate experiment
        for seed_idx in range(num_seeds):
            current_seed = start_seed + seed_idx
            logger.info(f"Running with seed {current_seed} ({seed_idx+1}/{num_seeds})")

            if mock:
                # Skip config creation and use mock metrics
                logger.info("Using mock mode - skipping Hydra config creation")
                metrics = run_algorithm_with_config(None, None, mock=True, seed=current_seed)
            else:
                # Create config with overrides
                # Use a specific config file for the algorithm
                config_name = f"default_{algorithm}"

                cfg = create_hydra_config(
                    config_path=f"../../../configs/default/{arch}",
                    config_name=config_name,
                    overrides=overrides,
                )

                # Run algorithm with config
                metrics = run_algorithm_with_config(cfg, module_path, mock=False, seed=current_seed)

            # Add this seed run to our results
            result.add_seed_run(current_seed, metrics)

        # Get baseline if needed
        if use_baseline and not establish_baseline:
            baseline_data = load_baseline(algorithm, environment)

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
            if save_baseline(
                algorithm, environment, result.metrics, result.metric_stats, result.seed_metrics
            ):
                result.established_baseline = True
                result.message = f"Established new baseline for {algorithm} on {environment}"
                logger.info(result.message)
            else:
                result.message = f"Failed to establish baseline for {algorithm} on {environment}"
                logger.error(result.message)
                return result

        # Mark test as successful
        result.success = True

    except Exception as e:
        result.success = False
        result.message = f"Error: {str(e)}"
        logger.error(f"Test failed: {result.message}")
        logger.error(traceback.format_exc())

    # Log summary
    logger.info(f"Test result: {result.summary}")
    return result
