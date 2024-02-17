import chex
import jax.numpy as jnp
from flax import linen as nn

from stoix.types import Observation


class ObservationInput(nn.Module):
    """Only Observation Input."""

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        observation = observation.agent_view
        return observation


class ObservationActionInput(nn.Module):
    """Observation and Action Input."""

    @nn.compact
    def __call__(self, observation: Observation, action: chex.Array) -> chex.Array:
        observation = observation.agent_view
        x = jnp.concatenate([observation, action], axis=-1)
        return x
