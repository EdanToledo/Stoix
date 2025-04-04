from typing import TYPE_CHECKING, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from stoix.base_types import State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class RecordEpisodeMetricsState:
    """State of the `LogWrapper`."""

    env_state: State
    key: chex.PRNGKey
    # Temporary variables to keep track of the episode return and length.
    running_count_episode_return: chex.Numeric
    running_count_episode_length: chex.Numeric
    # Final episode return and length.
    episode_return: chex.Numeric
    episode_length: chex.Numeric


class RecordEpisodeMetrics(Wrapper):
    """Record the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment."""
        key, reset_key = jax.random.split(key)
        state, timestep = self._env.reset(reset_key)
        state = RecordEpisodeMetricsState(
            state,
            key,
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
        )
        timestep.extras["episode_metrics"] = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        return state, timestep

    def step(
        self,
        state: RecordEpisodeMetricsState,
        action: chex.Array,
    ) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        done = timestep.last()
        not_done = 1 - done

        # Counting episode return and length.
        new_episode_return = state.running_count_episode_return + timestep.reward
        new_episode_length = state.running_count_episode_length + 1

        # Previous episode return/length until done and then the next episode return.
        episode_return_info = state.episode_return * not_done + new_episode_return * done
        episode_length_info = state.episode_length * not_done + new_episode_length * done

        timestep.extras["episode_metrics"] = {
            "episode_return": episode_return_info,
            "episode_length": episode_length_info,
            "is_terminal_step": done,
        }

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            key=state.key,
            running_count_episode_return=new_episode_return * not_done,
            running_count_episode_length=new_episode_length * not_done,
            episode_return=episode_return_info,
            episode_length=episode_length_info,
        )
        return state, timestep


def get_final_step_metrics(metrics: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], bool]:
    """Get the metrics for the final step of an episode and check if there was a final step
    within the provided metrics.

    Note: this is not a jittable method. We need to return variable length arrays, since
    we don't know how many episodes have been run. This is done since the logger
    expects arrays for computing summary statistics on the episode metrics.
    """
    is_final_ep = metrics.pop("is_terminal_step")
    has_final_ep_step = bool(jnp.any(is_final_ep))

    final_metrics: Dict[str, chex.Array]
    # If it didn't make it to the final step, return zeros.
    if not has_final_ep_step:
        final_metrics = jax.tree_util.tree_map(jnp.zeros_like, metrics)
    else:
        final_metrics = jax.tree_util.tree_map(lambda x: x[is_final_ep], metrics)

    return final_metrics, has_final_ep_step
