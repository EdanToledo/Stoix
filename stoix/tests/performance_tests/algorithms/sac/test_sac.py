"""
SAC (Soft Actor-Critic) Performance Tests

This module contains performance tests for the SAC algorithm implemented in Stoix.
Tests verify that the algorithm performs as expected on standard benchmark environments
and maintains performance relative to established baselines.
"""

import logging
from typing import Dict, Any, Optional

from stoix.tests.performance_tests.framework.registry import register_test
from stoix.tests.performance_tests.framework.utils import test_algorithm_performance

logger = logging.getLogger(__name__)

@register_test(algorithm="ff_sac", environment="brax/ant", module_path="stoix.systems.sac.ff_sac", arch="anakin")
def test_sac_ant(
    establish_baseline: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Test SAC performance on the Brax Ant environment.
    
    The Ant environment features a quadruped robot that must learn to walk forward
    as fast as possible. This is a standard benchmark for continuous control algorithms.
    
    SAC is well-suited for this task as it efficiently handles the continuous action space
    and exploration challenges.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        
    Returns:
        Dict with performance metrics and comparison to baseline.
    """
    
    # Environment-specific config overrides for optimal performance
    # These have been tuned based on prior experiments.
    all_overrides = {}
    
    # Apply user-provided overrides (these take precedence over defaults)
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_sac",
        environment="brax/ant",
        module_path="stoix.systems.sac.ff_sac",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides
    )

@register_test(algorithm="ff_sac", environment="brax/humanoid", module_path="stoix.systems.sac.ff_sac", arch="anakin")
def test_sac_humanoid(
    establish_baseline: bool = False,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Test SAC performance on the Brax Humanoid environment.
    
    The Humanoid environment is a complex humanoid robot that must learn to walk.
    This is one of the most challenging continuous control benchmarks due to its
    high-dimensional state and action spaces.
    
    SAC is well-suited for this task as it can handle the complexity and benefits
    from exploration through entropy maximization.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        max_steps: Maximum number of training steps.
        config_overrides: Dictionary of config overrides.
        
    Returns:
        Dict with performance metrics and comparison to baseline.
    """
    # Environment-specific config overrides for optimal performance
    # These have been tuned based on prior experiments.
    all_overrides = {}
    
    # Apply user-provided overrides (these take precedence over defaults)
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_sac",
        environment="brax/humanoid",
        module_path="stoix.systems.sac.ff_sac",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides
    ) 