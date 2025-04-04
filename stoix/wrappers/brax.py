from typing import Any, Dict, Optional, Tuple

import chex
import jax.numpy as jnp
from brax import base
from brax.envs.base import Wrapper as BraxWrapper
from flax import struct
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.types import StepType, TimeStep, restart

from stoix.base_types import Observation


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


class BraxJumanjiWrapper(BraxWrapper):
    def __init__(
        self,
        env: Environment,
    ):
        """Initialises a Brax wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self._env = env
        self._action_dim = self.action_spec().shape[0]

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
            step_count=jnp.array(0, dtype=int),
        )
        agent_view = new_state.obs.astype(float)
        legal_action_mask = jnp.ones((self._action_dim,), dtype=float)

        timestep = restart(
            observation=Observation(
                agent_view,
                legal_action_mask,
                new_state.step_count,
            ),
            extras={},
        )

        return new_state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        # If the previous step was truncated
        prev_truncated = state.info["truncation"].astype(jnp.bool_)
        # If the previous step was done
        prev_terminated = state.done.astype(jnp.bool_)

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
        # If terminated make the discount zero, otherwise one
        discount = jnp.where(terminated, 0.0, 1.0)
        # However, if truncated, make the discount one
        discount = jnp.where(truncated, 1.0, discount)
        # Lastly, if the previous step was truncated or terminated, make the discount zero
        # This is to ensure that the discount is zero for the last step of the episode
        # and that stepping past the last step of the episode does not affect the discount
        discount = jnp.where(prev_truncated | prev_terminated, 0.0, discount)

        # If terminated or truncated step type is last, otherwise mid
        step_type = jnp.where(terminated | truncated, StepType.LAST, StepType.MID)

        agent_view = state.obs.astype(float)
        legal_action_mask = jnp.ones((self._action_dim,), dtype=float)
        obs = Observation(
            agent_view,
            legal_action_mask,
            state.step_count,
        )

        next_timestep = TimeStep(
            step_type=step_type,
            reward=state.reward.astype(float),
            discount=discount.astype(float),
            observation=obs,
            extras={},
        )

        return state, next_timestep

    def action_spec(self) -> specs.Spec:
        action_space = specs.BoundedArray(
            shape=(self.action_size,),
            dtype=float,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )
        return action_space

    def observation_spec(self) -> specs.Spec:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=specs.Array(shape=(self.observation_size,), dtype=float),
            action_mask=specs.Array(shape=(self.action_size,), dtype=float),
            step_count=specs.Array(shape=(), dtype=int),
        )

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount")
