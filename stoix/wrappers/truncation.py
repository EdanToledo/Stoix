from typing import Tuple

import chex
import jax
from jumanji.env import State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from stoix.types import Observation


# TODO (edan): Verify
class TruncationAutoResetWrapper(Wrapper):
    """Wrapper that automatically resets the environment when the episode ends and places the
    final observation in the extras field of the timestep."""

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = self._env.reset(key)
        timestep = self._obs_in_extras(timestep)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:

        state, timestep = self._env.step(state, action)
        state, timestep = self._maybe_reset(state, timestep)

        return state, timestep

    def _auto_reset(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep[Observation]]:
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)
        state, reset_timestep = self._env.reset(key)

        extras = timestep.extras
        extras["final_observation"] = timestep.observation

        # Replace observation with reset observation.
        timestep = timestep.replace(observation=reset_timestep.observation, extras=extras)  # type: ignore

        return state, timestep

    def _obs_in_extras(self, timestep: TimeStep[Observation]) -> TimeStep[Observation]:
        extras = timestep.extras
        extras["final_observation"] = timestep.observation
        return timestep.replace(extras=extras)

    def _maybe_reset(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda st, ts: (st, self._obs_in_extras(ts)),
            state,
            timestep,
        )

        return state, timestep
