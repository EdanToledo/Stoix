from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.specs import Array, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from stoix.base_types import Observation


class FlattenObservationWrapper(Wrapper):
    """Simple wrapper that flattens the agent view observation."""

    def __init__(self, env: Environment) -> None:
        self._env = env
        obs_shape = self._env.observation_spec().agent_view.shape
        self._obs_shape = (np.prod(obs_shape),)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        agent_view = timestep.observation.agent_view.astype(jnp.float32)
        agent_view = agent_view.reshape(self._obs_shape)
        timestep = timestep.replace(
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        agent_view = timestep.observation.agent_view.astype(jnp.float32)
        agent_view = agent_view.reshape(self._obs_shape)
        timestep = timestep.replace(
            observation=timestep.observation._replace(agent_view=agent_view),
        )
        return state, timestep

    def observation_spec(self) -> Spec:
        return self._env.observation_spec().replace(
            agent_view=Array(shape=self._obs_shape, dtype=jnp.float32)
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
