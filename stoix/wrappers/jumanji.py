from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.specs import Array, MultiDiscreteArray, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import MultiToSingleWrapper, Wrapper

from stoix.base_types import Observation
from stoix.wrappers.transforms import MultiDiscreteToDiscrete


class JumanjiWrapper(Wrapper):
    def __init__(
        self,
        env: Environment,
        observation_attribute: str,
        multi_agent: bool = False,
    ) -> None:
        if isinstance(env.action_spec(), MultiDiscreteArray):
            env = MultiDiscreteToDiscrete(env)
        if multi_agent:
            env = MultiToSingleWrapper(env)

        self._env = env
        self._observation_attribute = observation_attribute
        self._legal_action_mask = jnp.ones((self.action_spec().num_values,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        if self._observation_attribute:
            agent_view = timestep.observation._asdict()[self._observation_attribute].astype(
                jnp.float32
            )
        else:
            agent_view = timestep.observation
        obs = Observation(agent_view, self._legal_action_mask, state.step_count)
        timestep_extras = timestep.extras
        if not timestep_extras:
            timestep_extras = {}
        timestep = timestep.replace(
            observation=obs,
            extras=timestep_extras,
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        if self._observation_attribute:
            agent_view = timestep.observation._asdict()[self._observation_attribute].astype(
                jnp.float32
            )
        else:
            agent_view = timestep.observation
        obs = Observation(agent_view, self._legal_action_mask, state.step_count)
        timestep_extras = timestep.extras
        if not timestep_extras:
            timestep_extras = {}
        timestep = timestep.replace(
            observation=obs,
            extras=timestep_extras,
        )
        return state, timestep

    def observation_spec(self) -> Spec:
        if self._observation_attribute:
            agent_view_spec = Array(
                shape=self._env.observation_spec().__dict__[self._observation_attribute].shape,
                dtype=jnp.float32,
            )
        else:
            agent_view_spec = self._env.observation_spec()
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=Array(shape=self._legal_action_mask.shape, dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )
