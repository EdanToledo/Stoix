from typing import Sequence

import chex
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.layers import NoisyLinear
from stoix.networks.utils import parse_activation_fn


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(layer_size, kernel_init=self.kernel_init)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class NoisyMLPTorso(nn.Module):
    """MLP torso using NoisyLinear layers instead of standard Dense layers."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True
    sigma_zero: float = 0.5

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        x = observation
        for layer_size in self.layer_sizes:
            x = NoisyLinear(layer_size, sigma_zero=self.sigma_zero)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class CNNTorso(nn.Module):
    """2D CNN torso. Expects input of shape (batch, height, width, channels).
    After this torso, the output is flattened."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for channel, kernel, stride in zip(self.channel_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(channel, (kernel, kernel), (stride, stride))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = parse_activation_fn(self.activation)(x)

        return x.reshape(*observation.shape[:-3], -1)
