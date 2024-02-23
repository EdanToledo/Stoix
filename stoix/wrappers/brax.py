from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from brax import base
from brax.envs.base import Wrapper as BraxWrapper
from brax.envs.wrappers.training import AutoResetWrapper
from flax import struct
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.types import StepType, TimeStep, restart

from stoix.types import Observation


@struct.dataclass
class BraxState(base.Base):
    pipeline_state: Optional[base.State]
    obs: chex.Array
    reward: chex.Numeric
    done: chex.Numeric
    key: chex.PRNGKey
    step_count: chex.Array
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class AddFinalObservation(BraxWrapper):
    """Adds the observation to the info dict."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["final_obs"] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        info = state.info
        info["final_obs"] = state.obs
        return state.replace(info=info)


class BraxJumanjiWrapper(BraxWrapper):
    def __init__(
        self,
        env: Environment,
        auto_reset: bool = True,
    ):
        """Initialises a Brax wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self._env = AddFinalObservation(env)
        if auto_reset:
            self._env = AutoResetWrapper(self._env)
        self._legal_action_mask = jnp.ones((self.action_spec().shape[0],), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:

        state = self._env.reset(key)

        new_state = BraxState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            key=key,
            metrics=state.metrics,
            info=state.info,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        timestep = restart(
            observation=Observation(
                new_state.obs,
                self._legal_action_mask,
                new_state.step_count,
            ),
            extras={
                "final_observation": Observation(new_state.info["final_obs"], self._legal_action_mask, new_state.step_count)
            },
        )

        return new_state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:

        state = self._env.step(state, action)
        state = BraxState(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            key=state.key,
            metrics=state.metrics,
            info=state.info,
            step_count=state.step_count + 1,
        )
        # This is true only if truncated
        truncated = state.info["truncation"].astype(jnp.bool_)
        # This is true if truncated or done
        terminated = state.done.astype(jnp.bool_)
        # This makes discount zero if terminated but not truncated
        discount = 1 - terminated.astype(jnp.float32) + truncated.astype(jnp.float32)

        # If terminated or truncated step type is last, otherwise mid
        step_type = jnp.where(terminated, StepType.LAST, StepType.MID)

        next_timestep = TimeStep(
            step_type=step_type,
            reward=state.reward,
            discount=discount,
            observation=Observation(state.obs, self._legal_action_mask, state.step_count),
            extras={"final_observation": Observation(state.info["final_obs"], self._legal_action_mask, state.step_count)},
        )

        return state, next_timestep

    def action_spec(self) -> specs.Spec:
        action_space = specs.BoundedArray(
            shape=(self.action_size,),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )
        return action_space

    def observation_spec(self) -> specs.Spec:

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=specs.Array(shape=(self.observation_size,), dtype=jnp.float32),
            action_mask=specs.Array(shape=(self.action_size,), dtype=jnp.float32),
            step_count=specs.Array(shape=(), dtype=jnp.int32),
        )
