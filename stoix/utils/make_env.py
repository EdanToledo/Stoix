import copy
import dataclasses
from typing import Callable, Tuple

import gymnax
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import jumanji
import navix
import pgx
import popjym
import xminigrid
from brax.envs import _envs as brax_environments
from brax.envs import create as brax_make
from gymnax import registered_envs as gymnax_environments
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.environment import EnvParams as GymnaxEnvParams
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.registration import registered_envs as jaxmarl_environments
from jumanji.env import Environment
from jumanji.registration import _REGISTRY as JUMANJI_REGISTRY
from jumanji.specs import BoundedArray, MultiDiscreteArray
from jumanji.wrappers import AutoResetWrapper, MultiToSingleWrapper
from navix import registry as navix_registry
from omegaconf import DictConfig
from popjym.registration import REGISTERED_ENVS as POPJYM_REGISTRY
from xminigrid.registration import _REGISTRY as XMINIGRID_REGISTRY

from stoix.utils.debug_env import IdentityGame, SequenceGame
from stoix.utils.env_factory import EnvFactory, EnvPoolFactory, GymnasiumFactory
from stoix.wrappers import GymnaxWrapper, JumanjiWrapper, RecordEpisodeMetrics
from stoix.wrappers.brax import BraxJumanjiWrapper
from stoix.wrappers.jax_to_factory import JaxEnvFactory
from stoix.wrappers.jaxmarl import JaxMarlWrapper, MabraxWrapper, SmaxWrapper
from stoix.wrappers.navix import NavixWrapper
from stoix.wrappers.pgx import PGXWrapper
from stoix.wrappers.transforms import (
    AddStartFlagAndPrevAction,
    MultiBoundedToBounded,
    MultiDiscreteToDiscrete,
)
from stoix.wrappers.xminigrid import XMiniGridWrapper


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
    # Config generator and select the wrapper.

    # Create envs.
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))
    if "generator" in env_kwargs:
        generator = env_kwargs.pop("generator")
        generator = hydra.utils.instantiate(generator)
        env_kwargs["generator"] = generator
    env = jumanji.make(env_name, **env_kwargs)
    eval_env = jumanji.make(env_name, **env_kwargs)
    env, eval_env = JumanjiWrapper(
        env, config.env.observation_attribute, config.env.multi_agent
    ), JumanjiWrapper(
        eval_env,
        config.env.observation_attribute,
        config.env.multi_agent,
    )

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def _create_gymnax_env_instance(
    env_name: str, env_kwargs: dict
) -> Tuple[GymnaxEnvironment, GymnaxEnvParams]:
    """Helper function to create a single Gymnax env instance with proper kwarg handling.

    This is due to gymnax having both environment init arguments and environment
    parameters in the EnvParams object."""
    # Get default params to identify which kwargs are for init and which are for params
    _, default_params = gymnax.make(env_name)
    param_fields = {f.name for f in dataclasses.fields(default_params)}

    init_kwargs = {k: v for k, v in env_kwargs.items() if k not in param_fields}
    params_kwargs = {k: v for k, v in env_kwargs.items() if k in param_fields}

    # Create env, passing only potential init_kwargs
    env, env_params = gymnax.make(env_name, **init_kwargs)

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

    # Wrap environments
    env = GymnaxWrapper(env, env_params)
    eval_env = GymnaxWrapper(eval_env, eval_env_params)

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
    # Config generator and select the wrapper.
    # Create envs.

    env, env_params = xminigrid.make(env_name, **config.env.kwargs)

    eval_env, eval_env_params = xminigrid.make(env_name, **config.env.kwargs)

    env = XMiniGridWrapper(env, env_params)
    eval_env = XMiniGridWrapper(eval_env, eval_env_params)

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
    # Config generator and select the wrapper.
    # Create envs.

    env = brax_make(env_name, auto_reset=False, **config.env.kwargs)

    eval_env = brax_make(env_name, auto_reset=False, **config.env.kwargs)

    env = BraxJumanjiWrapper(env)
    eval_env = BraxJumanjiWrapper(eval_env)

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_jaxmarl_env(
    env_name: str,
    config: DictConfig,
) -> Tuple[Environment, Environment]:
    """
     Create a JAXMARL environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A JAXMARL environment.
    """
    _jaxmarl_wrappers = {"Smax": SmaxWrapper, "MaBrax": MabraxWrapper}

    kwargs = dict(config.env.kwargs)
    if "smax" in env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Create jaxmarl envs.
    env = _jaxmarl_wrappers.get(config.env.env_name, JaxMarlWrapper)(
        jaxmarl.make(env_name, **kwargs),
        config.env.add_global_state,
        config.env.add_agent_ids_to_state,
    )
    eval_env = _jaxmarl_wrappers.get(config.env.env_name, JaxMarlWrapper)(
        jaxmarl.make(env_name, **kwargs),
        config.env.add_global_state,
        config.env.add_agent_ids_to_state,
    )
    env = MultiToSingleWrapper(env, reward_aggregator=jnp.mean)
    eval_env = MultiToSingleWrapper(eval_env, reward_aggregator=jnp.mean)

    if isinstance(env.action_spec(), MultiDiscreteArray):
        env = MultiDiscreteToDiscrete(env)
        eval_env = MultiDiscreteToDiscrete(eval_env)
    elif isinstance(env.action_spec(), BoundedArray):
        env = MultiBoundedToBounded(env)
        eval_env = MultiBoundedToBounded(eval_env)
    else:
        raise ValueError(f"Unsupported action spec for JAXMarl {env.action_spec()}.")

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_kinetix_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a Kinetix environment for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
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

    from stoix.wrappers.kinetix import KinetixWrapper

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

        return KinetixWrapper(env, env_params)

    env = _make_env(reset_fn=reset_fn_train, static_env_params=static_env_params_train)
    eval_env = _make_env(reset_fn=reset_fn_eval, static_env_params=static_env_params_eval)
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

    # Config generator and select the wrapper.
    craftax_environments = {
        "Craftax-Classic-Symbolic-v1": CraftaxClassicSymbolicEnv,
        "Craftax-Classic-Pixels-v1": CraftaxClassicPixelsEnv,
        "Craftax-Symbolic-v1": CraftaxSymbolicEnv,
        "Craftax-Pixels-v1": CraftaxPixelsEnv,
    }

    # Create envs.
    env = craftax_environments[env_name](**config.env.kwargs)
    eval_env = craftax_environments[env_name](**config.env.kwargs)

    env_params = env.default_params
    eval_env_params = eval_env.default_params

    env = GymnaxWrapper(env, env_params)
    eval_env = GymnaxWrapper(eval_env, eval_env_params)

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
    if "identity" in config.env.scenario.task_name.lower():
        env = IdentityGame(**config.env.kwargs)
        eval_env = IdentityGame(**config.env.kwargs)
    elif "sequence" in config.env.scenario.task_name.lower():
        env = SequenceGame(**config.env.kwargs)
        eval_env = SequenceGame(**config.env.kwargs)

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


def make_pgx_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a PGX environment for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """

    # Config generator and select the wrapper.
    # Create envs.
    env = pgx.make(env_name, **config.env.kwargs)
    eval_env = pgx.make(env_name, **config.env.kwargs)

    env = PGXWrapper(env)
    eval_env = PGXWrapper(eval_env)

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


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

    env = GymnaxWrapper(env, env_params)
    eval_env = GymnaxWrapper(eval_env, eval_env_params)

    env = AddStartFlagAndPrevAction(env)
    eval_env = AddStartFlagAndPrevAction(eval_env)

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

    env = NavixWrapper(env)
    eval_env = NavixWrapper(eval_env)

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_gymnasium_factory(
    env_name: str, config: DictConfig, apply_wrapper_fn: Callable
) -> GymnasiumFactory:

    env_factory = GymnasiumFactory(
        env_name, init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn, **config.env.kwargs
    )

    return env_factory


def make_envpool_factory(
    env_name: str, config: DictConfig, apply_wrapper_fn: Callable
) -> EnvPoolFactory:

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
    env_name = config.env.scenario.name

    if env_name in gymnax_environments:
        envs = make_gymnax_env(env_name, config)
    elif env_name in JUMANJI_REGISTRY:
        envs = make_jumanji_env(env_name, config)
    elif env_name in XMINIGRID_REGISTRY:
        envs = make_xland_minigrid_env(env_name, config)
    elif env_name in brax_environments:
        envs = make_brax_env(env_name, config)
    elif env_name in jaxmarl_environments:
        envs = make_jaxmarl_env(env_name, config)
    elif "craftax" in env_name.lower():
        envs = make_craftax_env(env_name, config)
    elif "kinetix" in env_name.lower():
        envs = make_kinetix_env(env_name, config)
    elif "debug" in env_name.lower():
        envs = make_debug_env(env_name, config)
    elif env_name in pgx.available_envs():
        envs = make_pgx_env(env_name, config)
    elif env_name in POPJYM_REGISTRY:
        envs = make_popjym_env(env_name, config)
    elif env_name in navix_registry():
        envs = make_navix_env(env_name, config)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")

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
    env_name = config.env.scenario.name
    suite_name = config.env.env_name

    apply_wrapper_fn = lambda x: x
    if "wrapper" in config.env and config.env.wrapper is not None:
        apply_wrapper_fn = hydra.utils.instantiate(config.env.wrapper, _partial_=True)

    if "envpool" in suite_name:
        return make_envpool_factory(env_name, config, apply_wrapper_fn)
    elif "gymnasium" in suite_name:
        return make_gymnasium_factory(env_name, config, apply_wrapper_fn)
    else:
        return JaxEnvFactory(
            make(config)[0], init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn
        )
