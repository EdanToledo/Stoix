import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class ArrayInput(nn.Module):
    """JAX Array Input. Used for any input that is already a JAX array."""

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return embedding


class FeatureInput(nn.Module):
    """Used for inputs that are specific attributes of some observation type."""

    feature_name: str

    @nn.compact
    def __call__(self, input_object: chex.ArrayTree) -> chex.Array:
        embedding = getattr(input_object, self.feature_name)
        return embedding


class EmbeddingActionInput(nn.Module):
    """Observation/Embedding and Action Input."""

    @nn.compact
    def __call__(self, embedding: chex.Array, action: chex.Array) -> chex.Array:
        """Concatenates observation/embedding and action."""
        x = jnp.concatenate([embedding, action], axis=-1)
        return x


class EmbeddingActionOnehotInput(nn.Module):
    """Observation/Embedding and Action One-hot Input."""

    action_dim: int

    @nn.compact
    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([observation_embedding, action_one_hot], axis=-1)
        return x
