import copy
from typing import Callable, Tuple

import hydra
from colorama import Fore, Style
from omegaconf import DictConfig
from stoa import Environment
from stoa import make_env as environments

from stoix.utils.env_factory import EnvFactory
from stoix.wrappers.jax_to_factory import JaxEnvFactory


def make(config: DictConfig) -> Tuple[Environment, Environment]:
    """Creates training and evaluation environments based on the provided configuration.

    This function uses a dispatcher to call the correct maker function for the
    specified environment suite (e.g., 'jumanji', 'brax'). This approach enables
    lazy loading of heavy dependencies, improving startup time.

    Args:
        config: The system configuration, which must specify `env.env_name`
            and `env.scenario.name`.

    Returns:
        A tuple containing the instantiated training and evaluation environments.
    """
    suite_name = config.env.env_name
    scenario_name = config.env.scenario.name
    env_kwargs = dict(copy.deepcopy(config.env.kwargs))

    # environment-specific kwargs modification
    if suite_name == "jumanji" and "generator" in env_kwargs:
        generator = env_kwargs.pop("generator")
        generator = hydra.utils.instantiate(generator)
        env_kwargs["generator"] = generator

    env = environments.make(suite_name, scenario_name, **env_kwargs)
    eval_env = environments.make(suite_name, scenario_name, **env_kwargs)

    wrapper = config.env.get("wrapper", None)
    if wrapper is not None:
        wrapper = hydra.utils.instantiate(wrapper, _partial_=True)
        env, eval_env = wrapper(env), wrapper(eval_env)

    env = environments.apply_core_wrappers(
        env,
        use_optimistic_reset=config.env.get("use_optimistic_reset", False),
        num_envs=config.arch.num_envs,
        reset_ratio=config.env.get("reset_ratio", 16),
        use_cached_auto_reset=config.env.get("use_cached_auto_reset", False),
    )

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Created environments for Suite: {suite_name} - "
        f"Scenario: {scenario_name}{Style.RESET_ALL}"
    )
    return env, eval_env


def make_factory(config: DictConfig) -> EnvFactory:
    """Creates a factory for generating environments.

    This is used for systems that require an environment factory rather than
    pre-instantiated environments, such as those using non-JAX environments
    like Gymnasium or EnvPool.

    Args:
        config: The system configuration.

    Returns:
        An `EnvFactory` instance.
    """
    suite_name = config.env.env_name
    scenario_name = config.env.scenario.name

    apply_wrapper_fn: Callable = lambda x: x
    if "wrapper" in config.env and config.env.wrapper is not None:
        apply_wrapper_fn = hydra.utils.instantiate(config.env.wrapper, _partial_=True)

    if suite_name == "envpool":
        from stoix.utils.env_factory import EnvPoolFactory

        return EnvPoolFactory(
            scenario_name,
            init_seed=config.arch.seed,
            apply_wrapper_fn=apply_wrapper_fn,
            **config.env.kwargs,
        )
    elif suite_name == "gymnasium":
        from stoix.utils.env_factory import GymnasiumFactory

        return GymnasiumFactory(
            scenario_name,
            init_seed=config.arch.seed,
            apply_wrapper_fn=apply_wrapper_fn,
            **config.env.kwargs,
        )
    else:
        # For all other JAX-based environments, create a single instance
        # and wrap it in a JaxEnvFactory.
        train_env = make(config)[0]
        return JaxEnvFactory(
            train_env, init_seed=config.arch.seed, apply_wrapper_fn=apply_wrapper_fn
        )
