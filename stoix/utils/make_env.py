import copy
import dataclasses
from typing import Callable, Tuple

import gymnax
import hydra
import jax
import jumanji
import jumanji.wrappers as jumanji_wrappers
import navix
import popgym_arcade
import popjym
import xminigrid
from brax.envs import _envs as brax_environments
from brax.envs import create as brax_make
from colorama import Fore, Style
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
    MultiDiscreteSpace,
    MultiDiscreteToDiscreteWrapper,
    NavixToStoa,
    ObservationExtractWrapper,
    RecordEpisodeMetrics,
    XMiniGridToStoa,
)
from stoa.core_wrappers.auto_reset import CachedAutoResetWrapper
from stoa.core_wrappers.optimistic_auto_reset import OptimisticResetVmapWrapper
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import AddRNGKey
from stoa.env_wrappers.kinetix import KinetixToStoa
from stoa.utility_wrappers.consistent_extras import ConsistentExtrasWrapper
from xminigrid.registration import _REGISTRY as XMINIGRID_REGISTRY

from stoix.utils.debug_env import DEBUG_ENVIRONMENTS
from stoix.utils.env_factory import EnvFactory, EnvPoolFactory, GymnasiumFactory
from stoix.wrappers.jax_to_factory import JaxEnvFactory


def apply_core_wrappers(env: Environment, config: DictConfig) -> Environment:
    """Apply core wrappers to the train environment.
    
    This includes wrappers for:
    - Adding RNG keys to environment state
    - Auto-resetting episodes when they terminate (unless using optimistic reset)
    - Recording episode metrics
    - Vectorization for efficient batched execution
    """
    
    # Always add RNG key support first (required by other wrappers)
    env = AddRNGKey(env)
    
    # Add episode metrics recording
    env = RecordEpisodeMetrics(env)
    
    # Choose between different reset and vectorization strategies
    if config.env.get("use_optimistic_reset", False):
        # OptimisticResetVmapWrapper handles both auto-reset and vectorization
        env = OptimisticResetVmapWrapper(env, config.arch.num_envs, min(config.env.get("reset_ratio", 16), config.arch.num_envs))
    else:
        # Apply separate auto-reset and vectorization wrappers
        
        # Add auto-reset functionality
        if config.env.get("use_cached_auto_reset", False):
            # Use cached auto-reset if available (more efficient)
            env = CachedAutoResetWrapper(env, next_obs_in_extras=True)
        else:
            # Use standard auto-reset
            env = AutoResetWrapper(env, next_obs_in_extras=True)
        
        # Add vectorization wrapper
        env = VmapWrapper(env)
    
    return env


def apply_optional_wrappers(
    envs: Tuple[Environment, Environment], config: DictConfig
) -> Tuple[Environment, Environment]:
    """Apply optional wrappers to the train and eval environments.

    Args:
        envs (Tuple[Environment, Environment]): The training and evaluation environments to wrap.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Unpack the environments.
    train_env, eval_env = envs

    # If a wrapper function is specified in the config, instantiate it.
    if "wrapper" in config.env and config.env.wrapper is not None:
        # Create the wrapper function.
        wrapper_fn = hydra.utils.instantiate(config.env.wrapper, _partial_=True)
        # Apply wrapper function to both environments.
        train_env = wrapper_fn(train_env)
        eval_env = wrapper_fn(eval_env)

    return train_env, eval_env


def make_jumanji_env(
    scenario_name: str,
    config: DictConfig,
) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
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
    env = jumanji.make(scenario_name, **env_kwargs)
    eval_env = jumanji.make(scenario_name, **env_kwargs)

    # If the environment is multi-agent, we convert it to a single-agent environment.
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

    # If the environment has a multi-discrete action space, we convert it to a single discrete action space.
    # This is the case for multi-agent jumanji environments.
    if isinstance(env.action_space(), MultiDiscreteSpace):
        env = MultiDiscreteToDiscreteWrapper(env)
        eval_env = MultiDiscreteToDiscreteWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def _create_gymnax_env_instance(
    scenario_name: str,
    env_kwargs: dict,
    env_make_fn: Callable[[str], Tuple[GymnaxEnvironment, GymnaxEnvParams]] = gymnax.make,
) -> Tuple[GymnaxEnvironment, GymnaxEnvParams]:
    """Helper function to create a single Gymnax (or gymnax-like) env instance with
    proper kwarg handling.

    This is due to gymnax having both environment init arguments and environment
    parameters in the EnvParams object."""
    # Get default params to identify which kwargs are for init and which are for params
    _, default_params = env_make_fn(scenario_name)
    param_fields = {f.name for f in dataclasses.fields(default_params)}

    init_kwargs = {k: v for k, v in env_kwargs.items() if k not in param_fields}
    params_kwargs = {k: v for k, v in env_kwargs.items() if k in param_fields}

    # Create env, passing only potential init_kwargs
    env, env_params = env_make_fn(scenario_name, **init_kwargs)

    # Update params with params_kwargs
    if params_kwargs:
        env_params = dataclasses.replace(env_params, **params_kwargs)

    return env, env_params


def make_gymnax_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a Gymnax environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))

    # Create envs using the helper function
    env, env_params = _create_gymnax_env_instance(scenario_name, env_kwargs)
    eval_env, eval_env_params = _create_gymnax_env_instance(scenario_name, env_kwargs)

    # Convert Gymnax environments to Stoa interface.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_popgym_arcade_env(
    scenario_name: str, config: DictConfig
) -> Tuple[Environment, Environment]:
    """
    Create a PopGym Arcade environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))

    # Create envs using the helper function
    env, env_params = _create_gymnax_env_instance(scenario_name, env_kwargs, popgym_arcade.make)
    eval_env, eval_env_params = _create_gymnax_env_instance(
        scenario_name, env_kwargs, popgym_arcade.make
    )

    # Convert Popgym Arcade environments to Stoa interface. These environments are based on Gymnax.
    # So we can use the GymnaxToStoa wrapper.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_xland_minigrid_env(
    scenario_name: str, config: DictConfig
) -> Tuple[Environment, Environment]:
    """
    Create a XLand Minigrid environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env, env_params = xminigrid.make(scenario_name, **config.env.kwargs)
    eval_env, eval_env_params = xminigrid.make(scenario_name, **config.env.kwargs)

    # Convert XLand Minigrid environments to Stoa interface.
    env = XMiniGridToStoa(env, env_params)
    eval_env = XMiniGridToStoa(eval_env, eval_env_params)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_brax_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a brax environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env = brax_make(scenario_name, auto_reset=False, **config.env.kwargs)
    eval_env = brax_make(scenario_name, auto_reset=False, **config.env.kwargs)

    # Convert Brax environments to Stoa interface.
    env = BraxToStoa(env)
    eval_env = BraxToStoa(eval_env)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_kinetix_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a Kinetix environment for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    from kinetix.environment import EnvState, StaticEnvParams, make_kinetix_env
    from kinetix.environment.env import KinetixEnv
    from kinetix.environment.ued.ued import make_reset_fn_sample_kinetix_level
    from kinetix.environment.utils import ActionType, ObservationType
    from kinetix.util.config import generate_params_from_config
    from kinetix.util.saving import load_evaluation_levels

    env_params, override_static_env_params = generate_params_from_config(
        dict(config.env.kinetix.env_size)
        | {
            "dense_reward_scale": config.env.dense_reward_scale,
            "frame_skip": config.env.frame_skip,
        }
    )

    def _get_static_params_and_reset_fn(
        level_config: DictConfig,
    ) -> tuple[Callable, StaticEnvParams]:
        if level_config.mode == "list":
            levels = level_config.levels
            levels_to_reset_to, static_env_params = load_evaluation_levels(levels)

            def reset(rng: jax.Array) -> EnvState:
                rng, _rng = jax.random.split(rng)
                level_idx = jax.random.randint(_rng, (), 0, len(levels))
                sampled_level = jax.tree.map(lambda x: x[level_idx], levels_to_reset_to)

                return sampled_level

        elif level_config.mode == "random":
            return (
                make_reset_fn_sample_kinetix_level(env_params, override_static_env_params),
                override_static_env_params,
            )
        else:
            raise ValueError(f"Unsupported level mode: {level_config.mode}")
        return reset, static_env_params

    reset_fn_train, static_env_params_train = _get_static_params_and_reset_fn(
        config.env.kinetix.train
    )
    reset_fn_eval, static_env_params_eval = _get_static_params_and_reset_fn(config.env.kinetix.eval)

    def _make_env(reset_fn: Callable, static_env_params: StaticEnvParams) -> KinetixEnv:

        env = make_kinetix_env(
            action_type=ActionType.from_string(config.env.scenario.action_type),
            observation_type=ObservationType.from_string(config.env.scenario.observation_type),
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
            auto_reset=False,
        )

        return KinetixToStoa(env, env_params)

    env = _make_env(reset_fn=reset_fn_train, static_env_params=static_env_params_train)
    eval_env = _make_env(reset_fn=reset_fn_eval, static_env_params=static_env_params_eval)
    
    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    # env = ConsistentExtrasWrapper(env)
    # eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_craftax_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a craftax environment for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
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
    env = craftax_environments[scenario_name](**config.env.kwargs)
    eval_env = craftax_environments[scenario_name](**config.env.kwargs)
    # Extract the default parameters from the environment.
    env_params = env.default_params
    eval_env_params = eval_env.default_params

    # Convert Craftax environments to Stoa interface.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_debug_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a debug environment for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create debug environments
    env = DEBUG_ENVIRONMENTS[scenario_name](**config.env.kwargs)
    eval_env = DEBUG_ENVIRONMENTS[scenario_name](**config.env.kwargs)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_popjym_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create POPJym environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env, env_params = popjym.make(scenario_name, **config.env.kwargs)
    eval_env, eval_env_params = popjym.make(scenario_name, **config.env.kwargs)

    # Convert POPJym environments to Stoa interface.
    # Popjym follows the Gymnax interface, so we can use the GymnaxToStoa wrapper.
    env = GymnaxToStoa(env, env_params)
    eval_env = GymnaxToStoa(eval_env, eval_env_params)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Add wrappers for adding start flag and previous action.
    env = AddStartFlagAndPrevAction(env)
    eval_env = AddStartFlagAndPrevAction(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_navix_env(scenario_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create Navix environments for training and evaluation.

    Args:
        scenario_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Create envs.
    env = navix.make(scenario_name, **config.env.kwargs)
    eval_env = navix.make(scenario_name, **config.env.kwargs)

    # Convert Navix environments to Stoa interface.
    env = NavixToStoa(env)
    eval_env = NavixToStoa(eval_env)

    # Add wrapper to ensure all extras field objects
    # are consistent for JAX scanning/while loops.
    env = ConsistentExtrasWrapper(env)
    eval_env = ConsistentExtrasWrapper(eval_env)

    # Apply any additional wrappers specified in the config.
    env, eval_env = apply_optional_wrappers((env, eval_env), config)

    # Add wrappers for auto-resetting and recording episode metrics.
    env = apply_core_wrappers(env, config)

    return env, eval_env


def make_gymnasium_factory(
    scenario_name: str, config: DictConfig, apply_wrapper_fn: Callable
) -> GymnasiumFactory:
    """Create a GymnasiumFactory for the specified environment.

    Args:
        scenario_name (str): The name of the environment.
        config (Dict): The configuration for the environment.
        apply_wrapper_fn (Callable): A function to apply wrappers to the environment.

    Returns:
        GymnasiumFactory: The created GymnasiumFactory.
    """
    env_factory = GymnasiumFactory(
        scenario_name,
        init_seed=config.arch.seed,
        apply_wrapper_fn=apply_wrapper_fn,
        **config.env.kwargs,
    )

    return env_factory


def make_envpool_factory(
    scenario_name: str, config: DictConfig, apply_wrapper_fn: Callable
) -> EnvPoolFactory:
    """Create an EnvPoolFactory for the specified environment.
    Args:
        scenario_name (str): The name of the environment.
        config (Dict): The configuration for the environment.
        apply_wrapper_fn (Callable): A function to apply wrappers to the environment.
    Returns:
        EnvPoolFactory: The created EnvPoolFactory.
    """
    env_factory = EnvPoolFactory(
        scenario_name,
        init_seed=config.arch.seed,
        apply_wrapper_fn=apply_wrapper_fn,
        **config.env.kwargs,
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
    suite_name = config.env.env_name
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
    elif "kinetix" in scenario_name.lower():
        envs = make_kinetix_env(scenario_name, config)
    else:
        raise ValueError(f"{scenario_name} is not a supported environment.")

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Created environments for Suite:{suite_name} - Scenario:{scenario_name}{Style.RESET_ALL}"
    )
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
