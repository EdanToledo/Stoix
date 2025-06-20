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
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin",
)
def test_ppo_ant(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Brax Ant environment.
    This is a standard continuous control benchmark.

    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = {
        "arch.total_num_envs": 1024,
        "system.rollout_length": 128,
        "system.epochs": 4,
        "system.num_minibatches": 16,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.vf_coef": 1.0,
        "system.ent_coef": 0.001,
        "system.max_grad_norm": 0.5,
        "arch.total_timesteps": 5e7,
        "arch.num_evaluation": 20,
    }

    if config_overrides:
        all_overrides.update(config_overrides)

    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="brax/ant",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds,
    )


@register_test(
    algorithm="ff_ppo",
    environment="brax/humanoid",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin",
)
def test_ppo_humanoid(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Brax Humanoid environment.
    This is a complex continuous control benchmark.

    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = {
        "arch.total_num_envs": 1024,
        "system.rollout_length": 128,
        "system.epochs": 4,
        "system.num_minibatches": 16,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.vf_coef": 1.0,
        "system.ent_coef": 0.001,
        "system.max_grad_norm": 0.5,
        "arch.total_timesteps": 1e8,
        "arch.num_evaluation": 20,
    }

    if config_overrides:
        all_overrides.update(config_overrides)

    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="brax/humanoid",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds,
    )


@register_test(
    algorithm="ff_ppo",
    environment="brax/halfcheetah",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin",
)
def test_ppo_halfcheetah(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Brax HalfCheetah environment.
    This is a standard continuous control benchmark.

    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = {
        "arch.total_num_envs": 1024,
        "system.rollout_length": 128,
        "system.epochs": 4,
        "system.num_minibatches": 16,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.vf_coef": 1.0,
        "system.ent_coef": 0.001,
        "system.max_grad_norm": 0.5,
        "arch.total_timesteps": 5e7,
        "arch.num_evaluation": 20,
    }

    if config_overrides:
        all_overrides.update(config_overrides)

    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="brax/halfcheetah",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds,
    )
