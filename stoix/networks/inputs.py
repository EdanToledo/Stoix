import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from stoix.base_types import Observation


class EmbeddingInput(nn.Module):
    """JAX Array Input."""

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return embedding


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


class EmbeddingActionInput(nn.Module):

    action_dim: int

    @nn.compact
    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        x = jnp.concatenate([observation_embedding, action], axis=-1)
        return x


class EmbeddingActionOnehotInput(nn.Module):

    action_dim: int

    @nn.compact
    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([observation_embedding, action_one_hot], axis=-1)
        return x
