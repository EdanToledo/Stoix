#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) Performance Tests

This module contains performance tests for the SAC algorithm implemented in Stoix.
Tests verify that the algorithm performs as expected on standard benchmark environments
and maintains performance relative to established baselines.

The tests in this module:
1. Test SAC on continuous control environments from the Brax suite
2. Apply appropriate hyperparameters for each environment
3. Compare performance metrics to established baselines

Each test can either run in evaluation mode (default) or establish new baselines
using the establish_baseline parameter.
"""

import logging
from typing import Dict, Any, Optional

from tests.performance_tests.framework.registry import register_test
from tests.performance_tests.framework.utils import test_algorithm_performance

logger = logging.getLogger(__name__)

@register_test(algorithm="ff_sac", environment="brax/ant")
def test_sac_ant(
    establish_baseline: bool = False,
    max_steps: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Test SAC performance on the Brax Ant environment.
    
    The Ant environment is a quadruped robot that must learn to walk forward.
    This is a medium-difficulty continuous control task that requires stable training.
    
    SAC typically performs well on this task due to its sample efficiency and
    ability to handle continuous action spaces.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        max_steps: Maximum number of training steps.
        config_overrides: Dictionary of config overrides.
        
    Returns:
        Dict with performance metrics and comparison to baseline.
    """
    # Environment-specific config overrides for optimal performance
    # These have been tuned based on prior experiments
    all_overrides = {
        # Use a larger buffer for better sample diversity
        "system.total_buffer_size": 1000000,
        # Learning rates optimized for Ant
        "system.actor_lr": 3e-4,
        "system.q_lr": 3e-4,
        "system.alpha_lr": 3e-4,
        # Batch size for efficient GPU utilization
        "system.total_batch_size": 256,
        # Standard reward scale for Ant environment
        "system.target_entropy_scale": 1.0,
        # Network architecture settings
        "network.actor_network.pre_torso.layer_sizes" : [256, 256],
    }
    
    # Apply user-provided overrides (these take precedence over defaults)
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_sac",
        environment="brax/ant",
        establish_baseline=establish_baseline,
        max_steps=max_steps,
        config_overrides=all_overrides
    )

@register_test(algorithm="ff_sac", environment="brax/humanoid")
def test_sac_humanoid(
    establish_baseline: bool = False,
    max_steps: Optional[int] = None,
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
    # These have been tuned based on prior experiments with Humanoid
    all_overrides = {
        # Use a larger buffer for better sample diversity
        "system.total_buffer_size": 1000000,
        # Learning rates optimized for Humanoid
        "system.actor_lr": 3e-4,
        "system.q_lr": 3e-4,
        "system.alpha_lr": 3e-4,
        # Batch size for efficient GPU utilization
        "system.total_batch_size": 256,
        # Longer warmup period due to task complexity
        "system.warmup_steps": 10000,
        # Temperature parameter tuning
        "system.init_alpha": 0.2,
        # Network architecture settings
        "network.actor_network.pre_torso.layer_sizes" : [256, 256],
        # Use a slightly lower discount to focus on shorter-term rewards
        "system.gamma": 0.99,
    }
    
    # Apply user-provided overrides (these take precedence over defaults)
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_sac",
        environment="brax/humanoid",
        establish_baseline=establish_baseline,
        max_steps=max_steps,
        config_overrides=all_overrides
    ) 