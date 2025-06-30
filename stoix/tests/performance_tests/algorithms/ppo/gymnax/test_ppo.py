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
    environment="gymnax/cartpole",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin",
)
def test_ppo_cartpole(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the CartPole environment.
    This is a simple environment that should converge quickly.

    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = {
        "arch.total_num_envs": 128,
        "system.rollout_length": 16,
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
        "arch.total_timesteps": 1e6,
        "arch.num_evaluation": 10,
    }

    if config_overrides:
        all_overrides.update(config_overrides)

    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="gymnax/cartpole",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds,
    )
