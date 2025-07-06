from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji.env import Environment
from jumanji.types import StepType, TimeStep, restart
from kinetix.util.saving import load_evaluation_levels
from omegaconf import DictConfig

from stoix.base_types import EvalResetFn, Observation, State
from stoix.wrappers.gymnax import GymnaxEnvState, GymnaxWrapper


class KinetixWrapper(GymnaxWrapper):
    """Puts the solve rate in the episode metrics, and allows for an override reset state."""

    def reset(
        self, key: chex.PRNGKey, override_state: Optional[State] = None
    ) -> Tuple[GymnaxEnvState, TimeStep]:
        key, reset_key = jax.random.split(key)
        obs, gymnax_state = self._env.reset(reset_key, self._env_params, override_state)
        obs = Observation(obs, self._legal_action_mask, jnp.array(0, dtype=int))
        timestep = restart(
            obs,
            extras={
                "episode_metrics": {
                    "solve_rate": jnp.array(0.0, dtype=float),  # Initialize solve rate
                    "distance": jnp.array(0.0, dtype=float),  # Initialize solve rate
                }
            },
        )
        state = GymnaxEnvState(
            key=key, gymnax_env_state=gymnax_state, step_count=jnp.array(0, dtype=int)
        )
        return state, timestep

    def step(self, state: GymnaxEnvState, action: chex.Array) -> Tuple[GymnaxEnvState, TimeStep]:
        key, key_step = jax.random.split(state.key)
        obs, gymnax_state, reward, done, info = self._env.step(
            key_step, state.gymnax_env_state, action, self._env_params
        )
        state = GymnaxEnvState(
            key=key, gymnax_env_state=gymnax_state, step_count=state.step_count + 1
        )

        timestep = TimeStep(
            observation=Observation(obs, self._legal_action_mask, state.step_count),
            reward=reward.astype(float),
            discount=jnp.array(1.0 - done, dtype=float),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
            extras={
                "episode_metrics": {
                    "solve_rate": info["GoalR"].astype(jnp.float32),  # Initialize solve rate
                    "distance": info["distance"],  # Initialize solve rate
                }
            },
        )
        return state, timestep


def make_kinetix_eval_reset_fn(config: DictConfig, env: Environment) -> EvalResetFn:
    """Creates a reset function that resets the environment to each of the specific evaluation levels.
    This assumes the number of evaluation environments must be a multiple of the number of levels.
    """
    levels = config.env.kinetix.eval.eval_levels
    levels_to_reset_to, _ = load_evaluation_levels(levels)

    def kinetix_eval_fn(key: chex.PRNGKey, num_environments: int) -> Tuple[State, TimeStep]:
        assert num_environments % len(levels) == 0, (
            f"Number of environments ({num_environments}) must be a multiple of the number of "
            f"evaluation levels ({len(levels)}). "
            f"Consider using a different value of `arch.num_eval_episodes`"
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

        states, timesteps = jax.vmap(
            env.reset,
        )(jnp.stack(env_keys), override_states)

        return states, timesteps

    return kinetix_eval_fn
