from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from kinetix.util.saving import load_evaluation_levels
from omegaconf import DictConfig
from stoa.env_types import TimeStep
from stoa.environment import Environment

from stoix.base_types import EvalResetFn, State
from stoix.evaluator import make_random_initial_eval_reset_fn


def make_kinetix_eval_reset_fn(config: DictConfig, env: Environment) -> EvalResetFn:
    """Creates a reset function that resets the environment to each of the specific evaluation levels.
    This assumes the number of evaluation environments must be a multiple of the number of levels.
    """
    if config.env.kinetix.eval.mode == "list":
        levels = config.env.kinetix.eval.levels
        levels_to_reset_to, _ = load_evaluation_levels(levels)

        def kinetix_eval_fn(key: chex.PRNGKey, num_environments: int) -> Tuple[State, TimeStep]:
            assert num_environments % len(levels) == 0, (
                f"Number of environments ({num_environments}) must be a multiple of the number of "
                f"evaluation levels ({len(levels)}). "
                f"Consider using a different value of `arch.num_eval_episodes`"
            )
            num_repeats = num_environments // len(levels)

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

            states, timesteps = jax.vmap(
                env.reset,
            )(jnp.stack(env_keys), override_states)

            return states, timesteps

        return kinetix_eval_fn
    else:
        return make_random_initial_eval_reset_fn(config, env)
