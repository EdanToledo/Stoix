from typing import Sequence

import chex
import distrax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.torso import MLPTorso


class DuelingQNetwork(nn.Module):

    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:

        value = MLPTorso(
            (*self.layer_sizes, 1), self.activation, self.use_layer_norm, self.kernel_init
        )(inputs)
        advantages = MLPTorso(
            (*self.layer_sizes, self.action_dim),
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
        )(inputs)

        # Advantages have zero mean.
        advantages -= jnp.mean(advantages, axis=-1, keepdims=True)

        q_values = value + advantages

        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)
