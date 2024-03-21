from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import StepType, TimeStep, restart

from stoix.base_types import Observation


@dataclass
class GameState:
    step_count: int
    state: chex.Array
    key: chex.PRNGKey


class IdentityGame(Environment):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def reset(self, key: chex.PRNGKey) -> Tuple[GameState, TimeStep]:
        state_val = jax.random.randint(key, shape=(1,), minval=0, maxval=self.num_actions)
        state = GameState(state=state_val, key=key, step_count=jnp.array(0))
        obs = Observation(
            agent_view=state_val.astype(jnp.float32),
            action_mask=jnp.ones(self.num_actions),
            step_count=state.step_count,
        )
        timestep = restart(obs, extras={})
        return state, timestep

    def step(self, state: chex, action: chex.Array) -> Tuple[GameState, TimeStep]:

        reward = jnp.where(action == state.state, 1.0, 0.0).squeeze()
        state_key, rng_key = jax.random.split(state.key)
        state_val = jax.random.randint(rng_key, shape=(1,), minval=0, maxval=self.num_actions)
        state = GameState(state=state_val, key=state_key, step_count=state.step_count + 1)
        discount = jnp.where(state.step_count < 50, 1.0, 0.0)
        timestep = TimeStep(
            step_type=jnp.where(state.step_count < 50, StepType.MID, StepType.LAST),
            observation=Observation(
                agent_view=state_val.astype(jnp.float32),
                action_mask=jnp.ones(self.num_actions),
                step_count=state.step_count,
            ),
            reward=reward,
            discount=discount,
            extras={},
        )

        return state, timestep

    def action_spec(self) -> Spec:
        return DiscreteArray(num_values=self.num_actions, dtype=jnp.int32)

    def observation_spec(self) -> Spec:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=Array(shape=(1,), dtype=jnp.float32),
            action_mask=Array(shape=(self.action_spec().num_values,), dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )


class SequenceGame(Environment):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def reset(self, key: chex.PRNGKey) -> Tuple[GameState, TimeStep]:
        state_val = jax.random.randint(key, shape=(1,), minval=0, maxval=self.num_actions)
        state = GameState(state=state_val, key=key, step_count=jnp.array(0))
        obs = Observation(
            agent_view=state_val.astype(jnp.float32),
            action_mask=jnp.ones(self.num_actions),
            step_count=state.step_count,
        )
        timestep = restart(obs, extras={})
        return state, timestep

    def step(self, state: chex, action: chex.Array) -> Tuple[GameState, TimeStep]:

        reward = jnp.where(action == state.state, 1.0, 0.0).squeeze()
        state_key, _ = jax.random.split(state.key)
        state_val = (state.state + 1) % self.num_actions
        state = GameState(state=state_val, key=state_key, step_count=state.step_count + 1)
        discount = jnp.where(state.step_count < 50, 1.0, 0.0)
        timestep = TimeStep(
            step_type=jnp.where(state.step_count < 50, StepType.MID, StepType.LAST),
            observation=Observation(
                agent_view=state_val.astype(jnp.float32),
                action_mask=jnp.ones(self.num_actions),
                step_count=state.step_count,
            ),
            reward=reward,
            discount=discount,
            extras={},
        )

        return state, timestep

    def action_spec(self) -> Spec:
        return DiscreteArray(num_values=self.num_actions, dtype=jnp.int32)

    def observation_spec(self) -> Spec:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=Array(shape=(1,), dtype=jnp.float32),
            action_mask=Array(shape=(self.action_spec().num_values,), dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )
