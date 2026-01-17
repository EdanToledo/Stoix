"""Environment creation utilities for Stoix.

This is a minimal fork that supports only JAX-native environments:
- Gymnax
- Jumanji
- Debug environments

Removed environment support (not compatible with JAX 0.7+):
- EnvPool
- Brax
- Kinetix
- Craftax
- PopGym Arcade
- XLand-MiniGrid
- PopJym
- Navix
- MuJoCo Playground
"""

import copy
import dataclasses
from typing import Any, Callable, Tuple

import hydra
from colorama import Fore, Style
from omegaconf import DictConfig
from stoa import (
    AddStartFlagAndPrevAction,
    AutoResetWrapper,
    Environment,
    EpisodeStepLimitWrapper,
    MultiDiscreteSpace,
    MultiDiscreteToDiscreteWrapper,
    ObservationExtractWrapper,
    RecordEpisodeMetrics,
)
from stoa.core_wrappers.auto_reset import CachedAutoResetWrapper
from stoa.core_wrappers.optimistic_auto_reset import OptimisticResetVmapWrapper
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import AddRNGKey
from stoa.utility_wrappers.extras_transforms import NoExtrasWrapper

from stoix.wrappers.jax_to_factory import JaxEnvFactory


def apply_core_wrappers(env: Environment, config: DictConfig) -> Environment:
    """Applies core wrappers for JAX-based environments.

    This includes wrappers for:
    - Adding an RNG key to the environment state.
    - Automatically resetting episodes upon termination.
    - Recording episode metrics (return, length, etc.).
    - Vectorizing the environment for batched execution.

    Args:
        env: The environment to wrap.
        config: The system configuration.

    Returns:
        The wrapped environment.
    """
    env = AddRNGKey(env)
    env = RecordEpisodeMetrics(env)

    if config.env.get("use_optimistic_reset", False):
        env = OptimisticResetVmapWrapper(
            env,
            config.arch.num_envs,
            min(config.env.get("reset_ratio", 16), config.arch.num_envs),
            next_obs_in_extras=True,
        )
    else:
        if config.env.get("use_cached_auto_reset", False):
            env = CachedAutoResetWrapper(env, next_obs_in_extras=True)
        else:
            env = AutoResetWrapper(env, next_obs_in_extras=True)
        env = VmapWrapper(env)
    return env


def apply_optional_wrappers(
    envs: Tuple[Environment, Environment], config: DictConfig
) -> Tuple[Environment, Environment]:
    """Applies any user-defined optional wrappers from the configuration.

    Args:
        envs: A tuple containing the training and evaluation environments.
        config: The system configuration.

    Returns:
        A tuple of the potentially wrapped environments.
    """
    train_env, eval_env = envs
    if "wrapper" in config.env and config.env.wrapper is not None:
        wrapper_fn = hydra.utils.instantiate(config.env.wrapper, _partial_=True)
        train_env = wrapper_fn(train_env)
        eval_env = wrapper_fn(eval_env)
    return train_env, eval_env


def make_jumanji_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """Creates and wraps a Jumanji environment."""
    import jumanji
    import jumanji.wrappers as jumanji_wrappers
    from stoa.env_adapters.jumanji import JumanjiToStoa

    env_kwargs = dict(copy.deepcopy(config.env.kwargs))
    if "generator" in env_kwargs:
        generator = env_kwargs.pop("generator")
        generator = hydra.utils.instantiate(generator)
        env_kwargs["generator"] = generator

    env = jumanji.make(scenario_name, **env_kwargs)
    eval_env = jumanji.make(scenario_name, **env_kwargs)

    if config.env.multi_agent:
        env = jumanji_wrappers.MultiToSingleWrapper(env)
        eval_env = jumanji_wrappers.MultiToSingleWrapper(eval_env)

    env = JumanjiToStoa(env)
    env = ObservationExtractWrapper(env, config.env.observation_attribute)
    eval_env = JumanjiToStoa(eval_env)
    eval_env = ObservationExtractWrapper(eval_env, config.env.observation_attribute)

    if isinstance(env.action_space(), MultiDiscreteSpace):
        env = MultiDiscreteToDiscreteWrapper(env)
        eval_env = MultiDiscreteToDiscreteWrapper(eval_env)

    env, eval_env = apply_optional_wrappers((env, eval_env), config)
    env = apply_core_wrappers(env, config)
    return env, eval_env


def _create_gymnax_env_instance(
    scenario_name: str,
    env_kwargs: dict,
    env_make_fn: Callable[[str], Tuple[Any, Any]],
) -> Tuple[Any, Any]:
    """Instantiates a Gymnax-like environment, handling init and param kwargs."""
    _, default_params = env_make_fn(scenario_name)
    param_fields = {f.name for f in dataclasses.fields(default_params)}

    init_kwargs = {k: v for k, v in env_kwargs.items() if k not in param_fields}
    params_kwargs = {k: v for k, v in env_kwargs.items() if k in param_fields}

    env, env_params = env_make_fn(scenario_name, **init_kwargs)
    if params_kwargs:
        env_params = dataclasses.replace(env_params, **params_kwargs)
    return env, env_params


def make_gymnax_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """Creates and wraps a Gymnax environment."""
    import gymnax
    from stoa.env_adapters.gymnax import GymnaxToStoa

    env_kwargs = dict(copy.deepcopy(config.env.kwargs))
    env, env_params = _create_gymnax_env_instance(scenario_name, env_kwargs, gymnax.make)
    eval_env, eval_env_params = _create_gymnax_env_instance(scenario_name, env_kwargs, gymnax.make)

    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)
    env = NoExtrasWrapper(env)
    eval_env = NoExtrasWrapper(eval_env)
    env, eval_env = apply_optional_wrappers((env, eval_env), config)
    env = apply_core_wrappers(env, config)
    return env, eval_env


def make_debug_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """Creates a simple debug environment for testing purposes."""
    from stoix.utils.debug_env import DEBUG_ENVIRONMENTS

    env = DEBUG_ENVIRONMENTS[scenario_name](**config.env.kwargs)
    eval_env = DEBUG_ENVIRONMENTS[scenario_name](**config.env.kwargs)
    env, eval_env = apply_optional_wrappers((env, eval_env), config)
    env = apply_core_wrappers(env, config)
    return env, eval_env


# A dispatcher mapping environment suite names to their respective maker functions.
# Only JAX-native environments are supported in this minimal fork.
ENV_MAKERS = {
    "jumanji": make_jumanji_env,
    "gymnax": make_gymnax_env,
    "debug": make_debug_env,
}


def make(config: DictConfig) -> Tuple[Environment, Environment]:
    """Creates training and evaluation environments based on the provided configuration.

    This function uses a dispatcher to call the correct maker function for the
    specified environment suite (e.g., 'jumanji', 'gymnax'). This approach enables
    lazy loading of heavy dependencies, improving startup time.

    Args:
        config: The system configuration, which must specify `env.env_name`
            and `env.scenario.name`.

    Returns:
        A tuple containing the instantiated training and evaluation environments.
    """
    suite_name = config.env.env_name
    scenario_name = config.env.scenario.name

    if suite_name not in ENV_MAKERS:
        raise ValueError(
            f"Unsupported environment suite '{suite_name}'. "
            f"Available suites: {list(ENV_MAKERS.keys())}"
        )

    maker_function = ENV_MAKERS[suite_name]
    envs = maker_function(scenario_name, config)

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Created environments for Suite: {suite_name} - "
        f"Scenario: {scenario_name}{Style.RESET_ALL}"
    )
    return envs


def make_factory(config: DictConfig) -> JaxEnvFactory:
    """Creates a factory for generating environments.

    This is used for systems that require an environment factory rather than
    pre-instantiated environments. In this minimal fork, only JAX-based
    environments are supported.

    Args:
        config: The system configuration.

    Returns:
        A `JaxEnvFactory` instance.
    """
    suite_name = config.env.env_name

    apply_wrapper_fn: Callable = lambda x: x
    if "wrapper" in config.env and config.env.wrapper is not None:
        apply_wrapper_fn = hydra.utils.instantiate(config.env.wrapper, _partial_=True)

    if suite_name in ("envpool", "gymnasium"):
        raise ValueError(
            f"Environment suite '{suite_name}' is not supported in this minimal fork. "
            f"Only JAX-native environments are supported: {list(ENV_MAKERS.keys())}"
        )

    # For all JAX-based environments, create a single instance and wrap it in a JaxEnvFactory.
    train_env = make(config)[0]
    return JaxEnvFactory(
        train_env, init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn
    )
