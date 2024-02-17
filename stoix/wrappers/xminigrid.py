from typing import TYPE_CHECKING, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
from xminigrid.environment import Environment, EnvParams, State

from stoix.types import Observation

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class XMiniGridEnvState:
    key: chex.PRNGKey
    minigrid_state_timestep: State


class XMiniGridWrapper(Wrapper):
    def __init__(self, env: Environment, env_params: EnvParams, flatten_observation: bool = False):
        self._env = env
        self._env_params = env_params
        self._flatten_observation = flatten_observation

        self._legal_action_mask = jnp.ones((self.action_spec().num_values,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[XMiniGridEnvState, TimeStep]:
        key, reset_key = jax.random.split(key)
        minigrid_state_timestep = self._env.reset(self._env_params, reset_key)
        obs = minigrid_state_timestep.observation
        if self._flatten_observation:
            obs = obs.flatten()
        obs = Observation(obs, self._legal_action_mask, jnp.array(0))
        timestep = TimeStep(
            observation=obs,
            reward=jnp.array(0.0),
            discount=jnp.array(1.0),
            step_type=minigrid_state_timestep.step_type,
            extras={},
        )
        state = XMiniGridEnvState(key=key, minigrid_state_timestep=minigrid_state_timestep)
        return state, timestep

    def step(self, state: XMiniGridEnvState, action: chex.Array) -> Tuple[State, TimeStep]:
        minigrid_state_timestep = self._env.step(
            self._env_params, state.minigrid_state_timestep, action
        )
        obs = minigrid_state_timestep.observation
        if self._flatten_observation:
            obs = obs.flatten()
        obs = Observation(
            obs,
            self._legal_action_mask,
            minigrid_state_timestep.state.step_num,
        )
        timestep = TimeStep(
            observation=obs,
            reward=minigrid_state_timestep.reward,
            discount=minigrid_state_timestep.discount,
            step_type=minigrid_state_timestep.step_type,
            extras={},
        )
        state = XMiniGridEnvState(
            key=minigrid_state_timestep.state.key, minigrid_state_timestep=minigrid_state_timestep
        )
        return state, timestep

    def action_spec(self) -> Spec:
        return DiscreteArray(num_values=self._env.num_actions(self._env_params))

    def observation_spec(self) -> Spec:
        obs_shape = self._env.observation_shape(self._env_params)
        if self._flatten_observation:
            obs_shape = (np.prod(obs_shape),)
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=Array(shape=obs_shape, dtype=jnp.float32),
            action_mask=Array(shape=(self.action_spec().num_values,), dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )
