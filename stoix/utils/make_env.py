import copy
import dataclasses
from typing import Callable, Tuple

import gymnax
import hydra
import jumanji
import jumanji.wrappers as jumanji_wrappers
import navix
import popgym_arcade
import popjym
import xminigrid
from brax.envs import _envs as brax_environments
from brax.envs import create as brax_make
from gymnax import registered_envs as gymnax_environments
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.environment import EnvParams as GymnaxEnvParams
from jumanji.registration import _REGISTRY as JUMANJI_REGISTRY
from navix import registry as navix_registry
from omegaconf import DictConfig
from popgym_arcade.registration import REGISTERED_ENVIRONMENTS as POPGYM_ARCADE_REGISTRY
from popjym.registration import REGISTERED_ENVS as POPJYM_REGISTRY
from stoa import (
    AddStartFlagAndPrevAction,
    AutoResetWrapper,
    BraxToStoa,
    Environment,
    GymnaxToStoa,
    JumanjiToStoa,
    NavixToStoa,
    ObservationExtractWrapper,
    RecordEpisodeMetrics,
    XMiniGridToStoa,
)
from xminigrid.registration import _REGISTRY as XMINIGRID_REGISTRY

from stoix.utils.debug_env import DEBUG_ENVIRONMENTS
from stoix.utils.env_factory import EnvFactory, EnvPoolFactory, GymnasiumFactory
from stoix.wrappers.jax_to_factory import JaxEnvFactory


def make_jumanji_env(
    env_name: str,
    config: DictConfig,
) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Set the generator if it exists in the config.
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))
    if "generator" in env_kwargs:
        generator = env_kwargs.pop("generator")
        generator = hydra.utils.instantiate(generator)
        env_kwargs["generator"] = generator

    # Create envs.
    env = jumanji.make(env_name, **env_kwargs)
    eval_env = jumanji.make(env_name, **env_kwargs)
    # If the environment is multi-agent, we need to wrap it to handle multiple agents.
    if config.env.multi_agent:
        env = jumanji_wrappers.MultiToSingleWrapper(env)
        eval_env = jumanji_wrappers.MultiToSingleWrapper(eval_env)

    # Convert Jumanji environments to Stoa interface.
    env = JumanjiToStoa(env)
    # Extract the observation attribute specified in the config.
    env = ObservationExtractWrapper(env, config.env.observation_attribute)

    # Do the same for the evaluation environment.
    eval_env = JumanjiToStoa(eval_env)
    eval_env = ObservationExtractWrapper(eval_env, config.env.observation_attribute)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def _create_gymnax_env_instance(
    env_name: str,
    env_kwargs: dict,
    env_make_fn: Callable[[str], Tuple[GymnaxEnvironment, GymnaxEnvParams]] = gymnax.make,
) -> Tuple[GymnaxEnvironment, GymnaxEnvParams]:
    """Helper function to create a single Gymnax (or gymnax-like) env instance with
    proper kwarg handling.

    This is due to gymnax having both environment init arguments and environment
    parameters in the EnvParams object."""
    # Get default params to identify which kwargs are for init and which are for params
    _, default_params = env_make_fn(env_name)
    param_fields = {f.name for f in dataclasses.fields(default_params)}

    init_kwargs = {k: v for k, v in env_kwargs.items() if k not in param_fields}
    params_kwargs = {k: v for k, v in env_kwargs.items() if k in param_fields}

    # Create env, passing only potential init_kwargs
    env, env_params = env_make_fn(env_name, **init_kwargs)

    # Update params with params_kwargs
    if params_kwargs:
        env_params = dataclasses.replace(env_params, **params_kwargs)

    return env, env_params


def make_gymnax_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a Gymnax environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))

    # Create envs using the helper function
    env, env_params = _create_gymnax_env_instance(env_name, env_kwargs)
    eval_env, eval_env_params = _create_gymnax_env_instance(env_name, env_kwargs)

    # Convert Gymnax environments to Stoa interface.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_popgym_arcade_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a PopGym Arcade environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))

    # Create envs using the helper function
    env, env_params = _create_gymnax_env_instance(env_name, env_kwargs, popgym_arcade.make)
    eval_env, eval_env_params = _create_gymnax_env_instance(
        env_name, env_kwargs, popgym_arcade.make
    )

    # Convert Popgym Arcade environments to Stoa interface. These environments are based on Gymnax.
    # So we can use the GymnaxToStoa wrapper.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_xland_minigrid_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a XLand Minigrid environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env, env_params = xminigrid.make(env_name, **config.env.kwargs)
    eval_env, eval_env_params = xminigrid.make(env_name, **config.env.kwargs)

    # Convert XLand Minigrid environments to Stoa interface.
    env = XMiniGridToStoa(env, env_params)
    eval_env = XMiniGridToStoa(eval_env, eval_env_params)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_brax_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a brax environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env = brax_make(env_name, auto_reset=False, **config.env.kwargs)
    eval_env = brax_make(env_name, auto_reset=False, **config.env.kwargs)

    # Convert Brax environments to Stoa interface.
    env = BraxToStoa(env)
    eval_env = BraxToStoa(eval_env)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_craftax_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a craftax environment for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    # We put the imports here so as to avoid the loading and processing of craftax
    # environments which happen in the imports
    from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
    from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
    from craftax.craftax_classic.envs.craftax_symbolic_env import (
        CraftaxClassicSymbolicEnv,
    )

    # Set up the environment mapping.
    craftax_environments = {
        "Craftax-Classic-Symbolic-v1": CraftaxClassicSymbolicEnv,
        "Craftax-Classic-Pixels-v1": CraftaxClassicPixelsEnv,
        "Craftax-Symbolic-v1": CraftaxSymbolicEnv,
        "Craftax-Pixels-v1": CraftaxPixelsEnv,
    }

    # Create envs.
    env = craftax_environments[env_name](**config.env.kwargs)
    eval_env = craftax_environments[env_name](**config.env.kwargs)
    # Extract the default parameters from the environment.
    env_params = env.default_params
    eval_env_params = eval_env.default_params
    # Convert Craftax environments to Stoa interface.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)
    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_debug_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a debug environment for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create debug environments
    env = DEBUG_ENVIRONMENTS[env_name](**config.env.kwargs)
    eval_env = DEBUG_ENVIRONMENTS[env_name](**config.env.kwargs)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def apply_optional_wrappers(
    envs: Tuple[Environment, Environment], config: DictConfig
) -> Tuple[Environment, Environment]:
    """Apply optional wrappers to the environments.

    Args:
        envs (Tuple[Environment, Environment]): The training and evaluation environments to wrap.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    envs = list(envs)
    if "wrapper" in config.env and config.env.wrapper is not None:
        for i in range(len(envs)):
            envs[i] = hydra.utils.instantiate(config.env.wrapper, env=envs[i])

    return tuple(envs)  # type: ignore


def make_popjym_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create POPJym environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env, env_params = popjym.make(env_name, **config.env.kwargs)
    eval_env, eval_env_params = popjym.make(env_name, **config.env.kwargs)

    # Convert POPJym environments to Stoa interface.
    # Popjym follows the Gymnax interface, so we can use the GymnaxToStoa wrapper.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrappers for adding start flag and previous action.
    env = AddStartFlagAndPrevAction(env)
    eval_env = AddStartFlagAndPrevAction(eval_env)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_navix_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create Navix environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env = navix.make(env_name, **config.env.kwargs)
    eval_env = navix.make(env_name, **config.env.kwargs)

    # Convert Navix environments to Stoa interface.
    env = NavixToStoa(env)
    eval_env = NavixToStoa(eval_env)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_gymnasium_factory(
    env_name: str, config: DictConfig, apply_wrapper_fn: Callable
) -> GymnasiumFactory:
    """Create a GymnasiumFactory for the specified environment.

    Args:
        env_name (str): The name of the environment.
        config (Dict): The configuration for the environment.
        apply_wrapper_fn (Callable): A function to apply wrappers to the environment.

    Returns:
        GymnasiumFactory: The created GymnasiumFactory.
    """
    env_factory = GymnasiumFactory(
        env_name, init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn, **config.env.kwargs
    )

    return env_factory


def make_envpool_factory(
    env_name: str, config: DictConfig, apply_wrapper_fn: Callable
) -> EnvPoolFactory:
    """Create an EnvPoolFactory for the specified environment.
    Args:
        env_name (str): The name of the environment.
        config (Dict): The configuration for the environment.
        apply_wrapper_fn (Callable): A function to apply wrappers to the environment.
    Returns:
        EnvPoolFactory: The created EnvPoolFactory.
    """
    env_factory = EnvPoolFactory(
        env_name, init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn, **config.env.kwargs
    )

    return env_factory


def make(config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create environments for training and evaluation.

    Args:
        config (Dict): The configuration of the environment.

    Returns:
        training and evaluation environments.
    """

    scenario_name = config.env.scenario.name

    if scenario_name in gymnax_environments:
        envs = make_gymnax_env(scenario_name, config)
    elif scenario_name in JUMANJI_REGISTRY:
        envs = make_jumanji_env(scenario_name, config)
    elif scenario_name in XMINIGRID_REGISTRY:
        envs = make_xland_minigrid_env(scenario_name, config)
    elif scenario_name in brax_environments:
        envs = make_brax_env(scenario_name, config)
    elif "craftax" in scenario_name.lower():
        envs = make_craftax_env(scenario_name, config)
    elif "debug" in scenario_name.lower():
        envs = make_debug_env(scenario_name, config)
    elif scenario_name in POPJYM_REGISTRY:
        envs = make_popjym_env(scenario_name, config)
    elif scenario_name in navix_registry():
        envs = make_navix_env(scenario_name, config)
    elif scenario_name in POPGYM_ARCADE_REGISTRY:
        envs = make_popgym_arcade_env(scenario_name, config)
    else:
        raise ValueError(f"{scenario_name} is not a supported environment.")

    envs = apply_optional_wrappers(envs, config)

    return envs


def make_factory(config: DictConfig) -> EnvFactory:
    """
    Create a env_factory for sebulba systems.

    Args:
        config (Dict): The configuration of the environment.

    Returns:
        A factory to create environments.
    """
    scenario_name = config.env.scenario.name
    suite_name = config.env.env_name

    apply_wrapper_fn = lambda x: x
    if "wrapper" in config.env and config.env.wrapper is not None:
        apply_wrapper_fn = hydra.utils.instantiate(config.env.wrapper, _partial_=True)

    if "envpool" in suite_name:
        return make_envpool_factory(scenario_name, config, apply_wrapper_fn)
    elif "gymnasium" in suite_name:
        return make_gymnasium_factory(scenario_name, config, apply_wrapper_fn)
    else:
        # For other environments, we use the JaxEnvFactory.
        # This factory will handle the creation of environments using Jumanji, Gymnax, etc.
        train_env = make(config)[0]
        return JaxEnvFactory(
            train_env, init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn
        )
