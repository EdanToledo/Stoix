from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from stoix.types import Observation

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

from chex import PRNGKey
from chex._src.pytypes import Array, ArrayTree
from jumanji import specs
from jumanji.types import StepType, TimeStep
from jumanji.wrappers import Wrapper
from pgx import Env


@dataclass(frozen=True)
class PGXState:
    env_state: ArrayTree
    key: PRNGKey


class PGXWrapper(Wrapper):
    def __init__(self, env: Env, max_episode_steps: int = 20_000) -> None:
        self._env = env
        self.max_episode_steps = max_episode_steps

    def reset(self, key: PRNGKey) -> Tuple[PGXState, TimeStep]:
        init_key, state_key = jax.random.split(key)
        init_state = self._env.init(init_key)
        state = PGXState(env_state=init_state, key=state_key)
        obs = Observation(
            agent_view=init_state.observation.astype(jnp.float32),
            action_mask=init_state.legal_action_mask.astype(jnp.float32),
            step_count=init_state._step_count,
        )
        reward = jnp.squeeze(init_state.rewards)
        discount = 1.0 - init_state.terminated.astype(jnp.float32).squeeze()
        timestep = TimeStep(
            observation=obs,
            reward=reward,
            discount=discount,
            step_type=StepType.FIRST,
            extras={"current_player": init_state.current_player},
        )
        return state, timestep

    def step(self, state: PGXState, action: Array) -> Tuple[PGXState, TimeStep]:
        new_step_key, new_state_key = jax.random.split(state.key)
        env_state = self._env.step(state.env_state, action, new_step_key)

        reward = jnp.squeeze(env_state.rewards)

        time_limit_reached = env_state._step_count >= self.max_episode_steps
        terminated = jnp.squeeze(env_state.terminated).astype(jnp.bool_) | time_limit_reached
        discount = 1.0 - terminated.astype(jnp.float32)

        step_type = jnp.where(
            terminated,
            StepType.LAST,
            StepType.MID,
        )
        obs = Observation(
            agent_view=env_state.observation.astype(jnp.float32),
            action_mask=env_state.legal_action_mask.astype(jnp.float32),
            step_count=env_state._step_count,
        )
        timestep = TimeStep(
            observation=obs,
            reward=reward,
            discount=discount,
            step_type=step_type,
            extras={"current_player": env_state.current_player},
        )
        new_state = PGXState(env_state=env_state, key=new_state_key)

        return new_state, timestep

    def action_spec(self) -> specs.Spec:
        """Returns the action spec."""
        action_space = specs.DiscreteArray(
            num_values=self._env.num_actions,
            dtype=jnp.int32,
            name="action",
        )
        return action_space

    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec."""
        agent_view_spec = specs.Array(shape=self._env.observation_shape, dtype=jnp.float32)
        action_mask_spec = specs.Array(shape=(self._env.num_actions,), dtype=jnp.float32)
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=action_mask_spec,
            step_count=specs.Array(shape=(), dtype=jnp.int32),
        )
