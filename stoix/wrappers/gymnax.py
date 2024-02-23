from typing import TYPE_CHECKING, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax import EnvParams, EnvState
from gymnax.environments.environment import Environment
from jumanji import specs
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from stoix.types import Observation

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class GymnaxEnvState:
    key: chex.PRNGKey
    gymnax_env_state: EnvState
    step_count: chex.Array


class GymnaxWrapper(Wrapper):
    def __init__(self, env: Environment, env_params: EnvParams):
        self._env = env
        self._env_params = env_params

        self._legal_action_mask = jnp.ones((self.action_spec().num_values,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[GymnaxEnvState, TimeStep]:
        key, reset_key = jax.random.split(key)
        obs, gymnax_state = self._env.reset(reset_key, self._env_params)
        obs = Observation(obs, self._legal_action_mask, jnp.array(0))
        timestep = restart(obs, extras={})
        state = GymnaxEnvState(key=key, gymnax_env_state=gymnax_state, step_count=jnp.array(0))
        return state, timestep

    def step(self, state: GymnaxEnvState, action: chex.Array) -> Tuple[GymnaxEnvState, TimeStep]:
        key, key_step = jax.random.split(state.key)
        obs, gymnax_state, reward, done, _ = self._env.step(
            key_step, state.gymnax_env_state, action, self._env_params
        )
        state = GymnaxEnvState(
            key=key, gymnax_env_state=gymnax_state, step_count=state.step_count + 1
        )

        timestep = TimeStep(
            observation=Observation(obs, self._legal_action_mask, state.step_count),
            reward=reward,
            discount=jnp.array(1.0 - done),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
            extras={},
        )
        return state, timestep

    def action_spec(self) -> Spec:
        return DiscreteArray(num_values=self._env.action_space(self._env_params).n)

    def observation_spec(self) -> Spec:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=Array(
                shape=self._env.observation_space(self._env_params).shape, dtype=jnp.float32
            ),
            action_mask=Array(shape=(self.action_spec().num_values,), dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )
