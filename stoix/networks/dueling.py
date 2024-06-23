from typing import Sequence

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.layers import NoisyLinear
from stoix.networks.torso import MLPTorso, NoisyMLPTorso


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
            (*self.layer_sizes, 1),
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
            activate_final=False,
        )(inputs)
        advantages = MLPTorso(
            (*self.layer_sizes, self.action_dim),
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
            activate_final=False,
        )(inputs)

        # Advantages have zero mean.
        advantages -= jnp.mean(advantages, axis=-1, keepdims=True)

        q_values = value + advantages

        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)


class DistributionalDuelingQNetwork(nn.Module):
    num_atoms: int
    vmax: float
    vmin: float
    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:

        value_torso = MLPTorso(
            self.layer_sizes, self.activation, self.use_layer_norm, self.kernel_init
        )(inputs)
        advantages_torso = MLPTorso(
            self.layer_sizes,
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
        )(inputs)

        value_logits = nn.Dense(self.num_atoms, kernel_init=self.kernel_init)(value_torso)
        value_logits = jnp.reshape(value_logits, (-1, 1, self.num_atoms))
        adv_logits = nn.Dense(self.action_dim * self.num_atoms, kernel_init=self.kernel_init)(
            advantages_torso
        )
        adv_logits = jnp.reshape(adv_logits, (-1, self.action_dim, self.num_atoms))
        q_logits = value_logits + adv_logits - adv_logits.mean(axis=1, keepdims=True)

        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = jnp.broadcast_to(atoms, (q_values.shape[0], self.num_atoms))
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_logits, atoms


class NoisyDistributionalDuelingQNetwork(nn.Module):
    num_atoms: int
    vmax: float
    vmin: float
    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    sigma_zero: float
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, embeddings: chex.Array) -> chex.Array:
        value_torso = NoisyMLPTorso(
            self.layer_sizes, self.activation, self.use_layer_norm, self.sigma_zero
        )(embeddings)
        advantages_torso = NoisyMLPTorso(
            self.layer_sizes, self.activation, self.use_layer_norm, self.sigma_zero
        )(embeddings)

        value_logits = NoisyLinear(self.num_atoms, sigma_zero=self.sigma_zero)(value_torso)
        value_logits = jnp.reshape(value_logits, (-1, 1, self.num_atoms))
        adv_logits = NoisyLinear(self.action_dim * self.num_atoms, sigma_zero=self.sigma_zero)(
            advantages_torso
        )
        adv_logits = jnp.reshape(adv_logits, (-1, self.action_dim, self.num_atoms))
        q_logits = value_logits + adv_logits - adv_logits.mean(axis=1, keepdims=True)

        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = jnp.broadcast_to(atoms, (q_values.shape[0], self.num_atoms))
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_logits, atoms
