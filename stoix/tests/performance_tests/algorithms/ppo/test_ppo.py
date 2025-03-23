"""
PPO (Proximal Policy Optimization) Performance Tests

This module contains performance tests for the PPO algorithm implemented in Stoix.
Tests verify that the algorithm performs as expected on standard benchmark environments
and maintains performance relative to established baselines.
"""
from stoix.tests.performance_tests.framework.registry import register_test
from stoix.tests.performance_tests.framework.utils import test_algorithm_performance

@register_test(
    algorithm="ff_ppo", 
    environment="brax/ant",
    module_path="stoix.systems.ppo.ff_ppo",
    arch="anakin"
)
def test_ppo_ant(establish_baseline=False, config_overrides=None):
    """
    Test PPO performance on the Brax Ant environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        
    Returns:
        TestResult object with performance metrics and comparison to baseline.
    """
    # Environment-specific config overrides for optimal performance
    # These have been tuned based on prior experiments.
    all_overrides = {}
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="brax/ant",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides
    )