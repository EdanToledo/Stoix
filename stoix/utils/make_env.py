import copy
from typing import Tuple

import gymnax
import hydra
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
from stoix.wrappers import GymnaxWrapper, JumanjiWrapper, RecordEpisodeMetrics
from stoix.wrappers.brax import BraxJumanjiWrapper
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


def make_gymnax_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a Gymnax environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    # Config generator and select the wrapper.
    # Create envs.
    env, env_params = gymnax.make(env_name, **config.env.kwargs)
    eval_env, eval_env_params = gymnax.make(env_name, **config.env.kwargs)

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
    if "wrapper" in config.env:
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


def make(config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create environments for training and evaluation..

    Args:
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
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
