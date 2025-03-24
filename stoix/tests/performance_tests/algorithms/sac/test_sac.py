"""
SAC (Soft Actor-Critic) Performance Tests

This module contains performance tests for the SAC algorithm implemented in Stoix.
Tests verify that the algorithm performs as expected on standard benchmark environments
and maintains performance relative to established baselines.
"""
from stoix.tests.performance_tests.framework.registry import register_test
from stoix.tests.performance_tests.framework.utils import test_algorithm_performance

@register_test(
    algorithm="ff_sac",
    environment="brax/ant",
    module_path="stoix.systems.sac.anakin.ff_sac",
    arch="anakin"
)
def test_sac_ant(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test SAC performance on the Brax Ant environment.
    This is a standard continuous control benchmark.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = {
        "arch.total_num_envs": 256,
        "system.rollout_length": 1,
        "system.epochs": 1,
        "system.warmup_steps": 16,
        "system.total_buffer_size": 25000,
        "system.total_batch_size": 256,
        "system.actor_lr": 3e-4,
        "system.q_lr": 3e-4,
        "system.alpha_lr": 3e-4,
        "system.tau": 0.005,
        "system.gamma": 0.99,
        "system.autotune": True,
        "system.target_entropy_scale": 1.0,
        "system.init_alpha": 0.1,
        "system.max_grad_norm": 0.5,
        "system.decay_learning_rates": False,
        "arch.total_timesteps": 1e6,
        "arch.num_evaluation": 10
    }
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_sac",
        environment="brax/ant",
        module_path="stoix.systems.sac.anakin.ff_sac",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_sac",
    environment="brax/halfcheetah",
    module_path="stoix.systems.sac.anakin.ff_sac",
    arch="anakin"
)
def test_sac_halfcheetah(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test SAC performance on the Brax HalfCheetah environment.
    This is a standard continuous control benchmark.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = {
        "arch.total_num_envs": 256,
        "system.rollout_length": 1,
        "system.epochs": 1,
        "system.warmup_steps": 16,
        "system.total_buffer_size": 25000,
        "system.total_batch_size": 256,
        "system.actor_lr": 3e-4,
        "system.q_lr": 3e-4,
        "system.alpha_lr": 3e-4,
        "system.tau": 0.005,
        "system.gamma": 0.99,
        "system.autotune": True,
        "system.target_entropy_scale": 1.0,
        "system.init_alpha": 0.1,
        "system.max_grad_norm": 0.5,
        "system.decay_learning_rates": False,
        "arch.total_timesteps": 1e6,
        "arch.num_evaluation": 10
    }
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_sac",
        environment="brax/halfcheetah",
        module_path="stoix.systems.sac.anakin.ff_sac",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    ) 