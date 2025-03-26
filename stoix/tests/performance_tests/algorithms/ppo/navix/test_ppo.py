"""
PPO (Proximal Policy Optimization) Performance Tests for Navix Environments

This module contains performance tests for the PPO algorithm implemented in Stoix
on Navix environments. Tests verify that the algorithm performs as expected on
standard benchmark environments and maintains performance relative to established baselines.
"""
from stoix.tests.performance_tests.framework.registry import register_test
from stoix.tests.performance_tests.framework.utils import test_algorithm_performance

# Common configuration for all tests
def get_base_config():
    """Returns the base configuration structure."""
    return {
        # "arch.total_timesteps": 1_048_576,
        "arch.total_timesteps": 5e4,
        "arch.num_evaluation": 1,
    }

@register_test(
    algorithm="ff_ppo",
    environment="navix/distshift2",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_distshift2(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix DistShift2 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_distshift2.yaml
    env_overrides = {
        "arch.total_num_envs": 256,
        "system.rollout_length": 64,
        "system.epochs": 16,
        "system.num_minibatches": 1,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 10,
        "system.gamma": 0.95,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/distshift2",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/doorkey_8x8",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_doorkey_8x8(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix DoorKey-8x8 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_doorkey_8x8.yaml
    env_overrides = {
        "arch.total_num_envs": 128,
        "system.rollout_length": 32,
        "system.epochs": 2,
        "system.num_minibatches": 32,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 10,
        "system.gamma": 0.95,
        "system.gae_lambda": 0.9,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/doorkey_8x8",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/doorkey_16x16",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_doorkey_16x16(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix DoorKey-16x16 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_doorkey_16x16.yaml
    env_overrides = {
        "arch.total_num_envs": 16,
        "system.rollout_length": 128,
        "system.epochs": 2,
        "system.num_minibatches": 1,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 1,
        "system.gamma": 0.95,
        "system.gae_lambda": 0.9,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/doorkey_16x16",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/dynamic_obstacles_6x6_random",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_dynamic_obstacles(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix Dynamic Obstacles environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Use similar settings to other grid navigation tasks
    env_overrides = {
        "arch.total_num_envs": 64,
        "system.rollout_length": 128,
        "system.epochs": 4,
        "system.num_minibatches": 4,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 1,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/dynamic_obstacles_6x6_random",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/empty_5x5",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_empty_5x5(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix Empty-5x5 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Use similar settings to empty_6x6
    env_overrides = {
        "arch.total_num_envs": 16,
        "system.rollout_length": 256,
        "system.epochs": 4,
        "system.num_minibatches": 16,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 1,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.9,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/empty_5x5",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/empty_6x6",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_empty_6x6(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix Empty-6x6 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_empty_6x6.yaml
    env_overrides = {
        "arch.total_num_envs": 16,
        "system.rollout_length": 256,
        "system.epochs": 4,
        "system.num_minibatches": 16,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 1,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.9,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/empty_6x6",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/empty_16x16",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_empty_16x16(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix Empty-16x16 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_empty_16x16.yaml
    env_overrides = {
        "arch.total_num_envs": 16,
        "system.rollout_length": 128,
        "system.epochs": 2,
        "system.num_minibatches": 1,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 1,
        "system.gamma": 0.95,
        "system.gae_lambda": 0.9,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/empty_16x16",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/empty_random_8x8",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_empty_random_8x8(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix Empty-Random-8x8 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_empty_random_8x8.yaml
    env_overrides = {
        "arch.total_num_envs": 64,
        "system.rollout_length": 128,
        "system.epochs": 8,
        "system.num_minibatches": 1,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 10,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.8,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/empty_random_8x8",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/fourrooms",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_fourrooms(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix FourRooms environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_fourrooms.yaml
    env_overrides = {
        "arch.total_num_envs": 128,
        "system.rollout_length": 32,
        "system.epochs": 2,
        "system.num_minibatches": 8,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 5,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.99,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/fourrooms",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/gotodoor_6x6",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_gotodoor_6x6(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix GoToDoor-6x6 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_gotodoor_6x6.yaml
    env_overrides = {
        "arch.total_num_envs": 32,
        "system.rollout_length": 32,
        "system.epochs": 8,
        "system.num_minibatches": 32,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 10,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/gotodoor_6x6",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/keycorridors4r4",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_keycorridors4r4(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix KeyCorridors-4Rooms environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_keycorridors4r3.yaml
    env_overrides = {
        "arch.total_num_envs": 128,
        "system.rollout_length": 64,
        "system.epochs": 4,
        "system.num_minibatches": 1,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 10,
        "system.gamma": 0.95,
        "system.gae_lambda": 0.95,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/keycorridors4r4",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/lavagaps6",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_lavagaps6(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix LavaGaps6 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_lavagap_s6.yaml
    env_overrides = {
        "arch.total_num_envs": 16,
        "system.rollout_length": 128,
        "system.epochs": 8,
        "system.num_minibatches": 8,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 0.5,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.99,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/lavagaps6",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    )

@register_test(
    algorithm="ff_ppo",
    environment="navix/simplecrossings9n1",
    module_path="stoix.systems.ppo.anakin.ff_ppo",
    arch="anakin"
)
def test_ppo_navix_simplecrossings9n1(establish_baseline=False, config_overrides=None, num_seeds=1):
    """
    Test PPO performance on the Navix SimpleCrossings-9N1 environment.
    
    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.
        num_seeds: Number of seeds to use for the test.
    """
    all_overrides = get_base_config()
    # Environment-specific overrides from navix_simplecrossings9n1.yaml
    env_overrides = {
        "arch.total_num_envs": 64,
        "system.rollout_length": 128,
        "system.epochs": 4,
        "system.num_minibatches": 4,
        "system.actor_lr": 3e-4,
        "system.critic_lr": 3e-4,
        "system.max_grad_norm": 0.5,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.9,
        "system.clip_eps": 0.2,
        "system.ent_coef": 0.01,
        "system.vf_coef": 0.5,
    }
    all_overrides.update(env_overrides)
    
    if config_overrides:
        all_overrides.update(config_overrides)
    
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="navix/simplecrossings9n1",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides,
        num_seeds=num_seeds
    ) 