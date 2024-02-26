from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.specs import Array, MultiDiscreteArray, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import MultiToSingleWrapper, Wrapper

from stoix.types import Observation


class JumanjiWrapper(Wrapper):
    def __init__(
        self,
        env: Environment,
        observation_attribute: str,
        flatten_observation: bool = False,
        multi_agent: bool = False,
    ) -> None:
        if isinstance(env.action_spec(), MultiDiscreteArray):
            env = MultiDiscreteToDiscrete(env)
        if multi_agent:
            env = MultiToSingleWrapper(env)

        self._env = env

        self._observation_attribute = observation_attribute
        self._flatten_observation = flatten_observation
        self._obs_shape = super().observation_spec().__dict__[self._observation_attribute].shape
        if self._flatten_observation:
            self._obs_shape = (np.prod(self._obs_shape),)
        self._legal_action_mask = jnp.ones((self.action_spec().num_values,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        obs = timestep.observation._asdict()[self._observation_attribute].astype(jnp.float32)
        timestep = timestep.replace(
            observation=Observation(
                obs.reshape(self._obs_shape), self._legal_action_mask, state.step_count
            ),
            extras={},
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        obs = timestep.observation._asdict()[self._observation_attribute].astype(jnp.float32)
        timestep = timestep.replace(
            observation=Observation(
                obs.reshape(self._obs_shape), self._legal_action_mask, state.step_count
            ),
            extras={},
        )
        return state, timestep

    def observation_spec(self) -> Spec:
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=Array(shape=self._obs_shape, dtype=jnp.float32),
            action_mask=Array(shape=(self.action_spec().num_values,), dtype=jnp.float32),
            step_count=Array(shape=(), dtype=jnp.int32),
        )


class MultiDiscreteToDiscrete(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._action_spec_num_values = env.action_spec().num_values

    def apply_factorisation(self, x: chex.Array) -> chex.Array:
        """Applies the factorisation to the given action."""
        action_components = []
        flat_action = x
        n = self._action_spec_num_values.shape[0]
        for i in range(n - 1, 0, -1):
            flat_action, remainder = jnp.divmod(flat_action, self._action_spec_num_values[i])
            action_components.append(remainder)
        action_components.append(flat_action)
        action = jnp.stack(
            list(reversed(action_components)),
            axis=-1,
            dtype=self._action_spec_num_values.dtype,
        )
        return action

    def inverse_factorisation(self, y: chex.Array) -> chex.Array:
        """Inverts the factorisation of the given action."""
        n = self._action_spec_num_values.shape[0]
        action_components = jnp.split(y, n, axis=-1)
        flat_action = action_components[0]
        for i in range(1, n):
            flat_action = self._action_spec_num_values[i] * flat_action + action_components[i]
        return flat_action

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        action = self.apply_factorisation(action)
        state, timestep = self._env.step(state, action)
        return state, timestep

    def action_spec(self) -> specs.Spec:
        """Returns the action spec of the environment."""
        original_action_spec = self._env.action_spec()
        num_actions = int(np.prod(np.asarray(original_action_spec.num_values)))
        return specs.DiscreteArray(num_actions, name="action")


class MultiBoundedToBounded(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._true_action_shape = env.action_spec().shape

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        action = action.reshape(self._true_action_shape)
        state, timestep = self._env.step(state, action)
        return state, timestep

    def action_spec(self) -> specs.Spec:
        """Returns the action spec of the environment."""
        original_action_spec = self._env.action_spec()
        size = int(np.prod(np.asarray(original_action_spec.shape)))
        return specs.BoundedArray(
            (size,),
            minimum=original_action_spec.minimum,
            maximum=original_action_spec.maximum,
            dtype=original_action_spec.dtype,
            name="action",
        )
