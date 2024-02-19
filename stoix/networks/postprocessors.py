import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class RescaleToSpec(nn.Module):
    minimum: float
    maximum: float

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:
        scale = self.maximum - self.minimum
        offset = self.minimum
        inputs = 0.5 * (inputs + 1.0)  # [0, 1]
        output = inputs * scale + offset  # [minimum, maximum]
        return output


class TanhToSpec(nn.Module):
    minimum: float
    maximum: float

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:
        scale = self.maximum - self.minimum
        offset = self.minimum
        inputs = jax.nn.tanh(inputs)  # [-1, 1]
        inputs = 0.5 * (inputs + 1.0)  # [0, 1]
        output = inputs * scale + offset  # [minimum, maximum]
        return output


class ClipToSpec(nn.Module):
    minimum: float
    maximum: float

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:
        output = jnp.clip(inputs, self.minimum, self.maximum)
        return output
