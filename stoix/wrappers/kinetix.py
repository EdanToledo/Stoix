from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.types import TimeStep, restart
from kinetix.util.saving import load_evaluation_levels
from omegaconf import DictConfig

from stoix.base_types import EvalResetFn, State
from stoix.wrappers.gymnax import GymnaxEnvState


def make_kinetix_eval_reset_fn(config: DictConfig, env: Environment) -> EvalResetFn:
    levels = config.env.kinetix.eval.eval_levels
    levels_to_reset_to, _ = load_evaluation_levels(levels)

    def kinetix_eval_fn(key: chex.PRNGKey, num_environments: int) -> Tuple[State, TimeStep]:
        print("OIN HERE", num_environments, levels)
        assert num_environments % len(levels) == 0, (
            f"Number of environments ({num_environments}) must be a multiple of the number of "
            f"evaluation levels ({len(levels)})."
        )
        num_repeats = num_environments // len(levels)
        print(jax.tree.map(lambda x: jnp.shape(x), levels_to_reset_to))
        override_states = jax.tree.map(
            lambda x: jnp.tile(
                x,
                [
                    num_repeats,
                ]
                + [1] * max(0, len(x.shape) - 1),
            ),
            levels_to_reset_to,
        )
        key, *env_keys = jax.random.split(key, num_environments + 1)
        obs, gymnax_states = jax.vmap(env._env.reset)(
            jnp.stack(env_keys), env.env_params, override_states
        )
        timesteps = restart(obs, extras={})
        states = GymnaxEnvState(
            key=key, gymnax_env_state=gymnax_states, step_count=jnp.array(0, dtype=int)
        )

        return states, timesteps

    return kinetix_eval_fn
