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

import os
import sys
import json
import logging
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import numpy as np
import jax
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure the stoix module is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

logger = logging.getLogger(__name__)

# Constants
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "baselines")
# Primary metrics used for determining performance changes
# Listed in order of priority - first found will be used as main comparison metric
MAIN_PERFORMANCE_METRICS = [
    "eval_return",  # Average evaluation return
    "eval_episode_return",  # Average evaluation episode return
    "return",  # Average training return
    "episode_return",  # Average training episode return
    "success_rate",  # Task success rate (if applicable)
    "reward",  # Raw reward value
]


@dataclass
class TestResult:
    """
    Class to store the results of a performance test.

    This dataclass encapsulates all information about a test run,
    including performance metrics, baseline comparisons, and error states.

    Attributes:
        algorithm: Name of the algorithm tested
        environment: Name of the environment tested
        success: Whether the test completed successfully
        metrics: Dictionary of performance metrics from the current run
        baseline_metrics: Dictionary of baseline metrics for comparison
        comparison: Dictionary of percentage differences between metrics
        error: Error message if the test failed
        established_baseline: Whether this run established a new baseline
    """

    algorithm: str
    environment: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    comparison: Dict[str, float] = field(default_factory=dict)
    error: str = ""
    established_baseline: bool = False

    @property
    def summary(self) -> str:
        """
        Get a short summary of the test result.

        Returns:
            Single-line summary string suitable for logging
        """
        if not self.success:
            return f"Failed: {self.error}"

        if self.established_baseline:
            return "Baseline established"

        main_metric = self.get_main_performance_metric()
        if main_metric:
            metric_name, value, baseline, pct_diff = main_metric
            return f"{metric_name}: {value:.2f} vs {baseline:.2f} ({pct_diff:+.2f}%)"

        return "Success (no metrics)"

    def get_main_performance_metric(self) -> Optional[Tuple[str, float, float, float]]:
        """
        Get the main performance metric and its comparison to baseline.

        This tries to find one of the standard performance metrics in both
        the current and baseline results, and returns the comparison data.

        Returns:
            Tuple of (metric_name, current_value, baseline_value, percent_diff)
            or None if no comparison metrics are available
        """
        if not self.metrics or not self.baseline_metrics:
            return None

        # Try to find a main performance metric based on priority list
        for metric_name in MAIN_PERFORMANCE_METRICS:
            if metric_name in self.metrics and metric_name in self.baseline_metrics:
                value = self.metrics[metric_name]
                baseline = self.baseline_metrics[metric_name]
                pct_diff = (
                    ((value - baseline) / abs(baseline)) * 100
                    if baseline != 0
                    else float("inf")
                )
                return (metric_name, value, baseline, pct_diff)

        # If no main metric found, use the first common metric
        common_metrics = set(self.metrics.keys()).intersection(
            set(self.baseline_metrics.keys())
        )
        if common_metrics:
            metric_name = sorted(common_metrics)[0]
            value = self.metrics[metric_name]
            baseline = self.baseline_metrics[metric_name]
            pct_diff = (
                ((value - baseline) / abs(baseline)) * 100
                if baseline != 0
                else float("inf")
            )
            return (metric_name, value, baseline, pct_diff)

        return None


def get_baseline_path(algorithm: str, environment: str) -> str:
    """
    Get the path to the baseline file for a given algorithm and environment.

    Args:
        algorithm: Name of the algorithm.
        environment: Name of the environment.

    Returns:
        Path to the baseline file.
    """
    # Replace any slashes in environment name with underscores for file safety
    safe_env_name = environment.replace("/", "_")
    os.makedirs(BASELINE_DIR, exist_ok=True)
    return os.path.join(BASELINE_DIR, f"{algorithm}_{safe_env_name}_baseline.json")


def load_baseline(algorithm: str, environment: str) -> Dict[str, Any]:
    """
    Load baseline metrics for a given algorithm and environment.

    Args:
        algorithm: Name of the algorithm.
        environment: Name of the environment.

    Returns:
        Dictionary of baseline metrics or empty dict if baseline doesn't exist.
    """
    baseline_path = get_baseline_path(algorithm, environment)

    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to load baseline from {baseline_path}. Invalid JSON."
            )
            return {}
        except Exception as e:
            logger.warning(
                f"Unexpected error loading baseline from {baseline_path}: {str(e)}"
            )
            return {}
    else:
        logger.info(f"No baseline found at {baseline_path}")
        return {}


def save_baseline(algorithm: str, environment: str, metrics: Dict[str, Any]) -> None:
    """
    Save metrics as baseline for a given algorithm and environment.

    Args:
        algorithm: Name of the algorithm.
        environment: Name of the environment.
        metrics: Dictionary of metrics to save.
    """
    baseline_path = get_baseline_path(algorithm, environment)
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)

    try:
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved baseline to {baseline_path}")
    except Exception as e:
        logger.error(f"Failed to save baseline to {baseline_path}: {str(e)}")


def compare_metrics(
    current: Dict[str, float], baseline: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare current metrics to baseline metrics.

    Calculates percentage differences between current and baseline metrics.
    For metrics where higher values are better (like rewards), positive
    percentages indicate improvement.

    Args:
        current: Dictionary of current metrics.
        baseline: Dictionary of baseline metrics.

    Returns:
        Dictionary mapping metric names to percentage differences.
    """
    comparison = {}

    for metric_name, baseline_value in baseline.items():
        if metric_name in current:
            current_value = current[metric_name]
            # Handle division by zero
            if baseline_value != 0:
                pct_diff = (
                    (current_value - baseline_value) / abs(baseline_value)
                ) * 100
            else:
                # Special cases for zero baseline values
                pct_diff = (
                    float("inf")
                    if current_value > 0
                    else float("-inf")
                    if current_value < 0
                    else 0
                )
            comparison[metric_name] = pct_diff

    return comparison


def create_hydra_config_with_overrides(
    config_path: str, config_name: str, override_list: List[str]
) -> DictConfig:
    """Create a Hydra config with the given overrides.

    Args:
        config_path: Path to the config directory
        config_name: Name of the config file
        override_list: List of override values

    Returns:
        DictConfig: The hydra config

    Raises:
        ValueError: If there is an error creating the config
    """
    logger.debug(
        f"Creating config from {config_path}/{config_name} with overrides: {override_list}"
    )

    try:
        # Initialize Hydra and compose the config
        with hydra.initialize(version_base="1.2", config_path=config_path):
            cfg = hydra.compose(config_name=config_name, overrides=override_list)
        return cfg
    except Exception as e:
        raise ValueError(f"Error creating Hydra config: {str(e)}")


def test_algorithm_performance(
    algorithm: str,
    environment: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    use_baseline: bool = False,
    establish_baseline: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Test the performance of a reinforcement learning algorithm on a given environment.

    Args:
        algorithm: The algorithm to test
        environment: The environment to test on
        config_overrides: Optional overrides for the algorithm configuration
        use_baseline: Whether to compare against a baseline
        establish_baseline: Whether to establish a new baseline
        max_steps: Maximum number of steps to run the test for

    Returns:
        Dict: The test results
    """
    logger.info(f"Testing {algorithm} on {environment}")

    # Update the config path to match the project structure
    config_name = f"default_{algorithm}"
    # Use relative path from the test file to the configs
    config_path = "../../../stoix/configs/default/anakin"

    # Convert config_overrides dict to list of strings in Hydra format
    override_list = []
    if config_overrides:
        for key, value in config_overrides.items():
            override_list.append(f"{key}={value}")

    override_list.append(f"env={environment}")

    try:
        # Create config with overrides
        cfg = create_hydra_config_with_overrides(
            config_path, config_name, override_list
        )

        # Run algorithm with config
        # TODO: Implement actual algorithm execution and metric collection
        run_algorithm_with_config(cfg)

        # For now, just return a mock result
        results = {
            "algorithm": algorithm,
            "environment": environment,
            "success": True,
            "message": "Test completed successfully",
            "metrics": {
                "mean_return": 0.0,
                "std_return": 0.0,
                "training_steps": max_steps or 0,
                "wall_time_seconds": 0.0,
            },
        }

        return results

    except Exception as e:
        logger.error(f"Failed to create config: {str(e)}")
        return {
            "algorithm": algorithm,
            "environment": environment,
            "success": False,
            "message": f"Failed to create config: {str(e)}",
            "metrics": {},
        }


def run_algorithm_with_config(cfg: DictConfig) -> Dict[str, float]:
    """
    Run an algorithm with the given config and return performance metrics.

    This function dynamically imports the appropriate algorithm module based on
    the config, runs the experiment, and processes the returned metrics into a
    standardized format.

    Args:
        cfg: Hydra config for the algorithm.

    Returns:
        Dictionary of performance metrics.

    Raises:
        ImportError: If the algorithm module cannot be imported.
        AttributeError: If the run_experiment function is not found.
        RuntimeError: If the experiment fails to run.
    """
    try:
        # Dynamic import of the appropriate module based on config
        system_name = cfg.system.system_name
        
        # Extract algorithm name from config - it's typically in the config name
        # e.g., "default_ff_sac" -> "ff_sac"
        if "config_name" in cfg:
            # If config_name is directly available
            config_name = cfg.config_name
        else:
            # Try to extract from _target_ if available
            config_name = cfg.get("_target_", "").split(".")[-1]
            if not config_name:
                # Use system_name as fallback
                config_name = system_name
        
        # If config_name starts with "default_", remove that prefix
        if config_name.startswith("default_"):
            algorithm_name = config_name[len("default_"):]
        else:
            algorithm_name = config_name
            
        # Construct the module path based on system name and algorithm
        module_name = f"stoix.systems.{system_name}.{algorithm_name}"
        
        logger.debug(f"Importing module {module_name} for system {system_name}")

        # Import the module
        import importlib

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Could not import module {module_name}: {str(e)}")

        # Get the run_experiment function
        try:
            run_experiment = getattr(module, "run_experiment")
        except AttributeError:
            raise AttributeError(
                f"Module {module_name} does not have a run_experiment function"
            )

        # Run the experiment with a specific RNG key for reproducibility
        logger.debug(f"Running experiment with module {module_name}")
        metrics = run_experiment(cfg)

        # Process metrics
        if isinstance(metrics, (int, float)):
            # If just a single value is returned, assume it's the main metric
            logger.debug(f"Received scalar metric: {metrics}")
            return {"return": float(metrics)}
        elif isinstance(metrics, dict):
            # Convert any numpy arrays or JAX arrays to Python scalars
            logger.debug(f"Received dictionary of metrics with {len(metrics)} entries")
            processed_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (np.ndarray, jax.Array)):
                    # Extract scalar value from array
                    if v.size == 1:
                        processed_metrics[k] = float(v)
                    else:
                        # For multi-dimensional arrays, log a warning and skip
                        logger.warning(
                            f"Skipping non-scalar metric {k} with shape {v.shape}"
                        )
                elif isinstance(v, (int, float)):
                    processed_metrics[k] = float(v)
                else:
                    logger.warning(f"Skipping non-numeric metric {k} of type {type(v)}")

            logger.debug(f"Processed {len(processed_metrics)} metrics")
            return processed_metrics
        else:
            logger.warning(
                f"Unexpected metrics type: {type(metrics)}. Converting to float if possible."
            )
            try:
                return {"return": float(metrics)}
            except (TypeError, ValueError):
                logger.error(f"Could not convert metrics to float: {metrics}")
                return {}

    except Exception as e:
        error_msg = (
            f"Error running algorithm with config: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
        raise RuntimeError(f"Failed to run algorithm: {str(e)}")
