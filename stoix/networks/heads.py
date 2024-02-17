from typing import Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, lecun_normal, orthogonal

from stoix.networks.distributions import TanhMultivariateNormalDiag


class CategoricalHead(nn.Module):
    action_dim: Union[int, Sequence[int]]
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.Categorical:

        logits = nn.Dense(np.prod(self.action_dim), kernel_init=self.kernel_init)(embedding)

        if not isinstance(self.action_dim, int):
            logits = logits.reshape(self.action_dim)

        return distrax.Categorical(logits=logits)


class TanhMultivariateNormalDiagHead(nn.Module):

    action_dim: int
    init_scale: float = 0.3
    min_scale: float = 1e-3
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> TanhMultivariateNormalDiag:

        loc = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        scale = jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding))

        scale *= self.init_scale / jax.nn.softplus(0.0)
        scale += self.min_scale

        return TanhMultivariateNormalDiag(loc=loc, scale_diag=scale)


class LinearOutputHead(nn.Module):
    action_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:

        x = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)

        return x


class ScalarCriticHead(nn.Module):
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(1, kernel_init=self.kernel_init)(embedding).squeeze(axis=-1)


class DiscreteQNetworkHead(nn.Module):
    action_dim: int
    epsilon: float = 0.1
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.EpsilonGreedy:

        q_values = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)

        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)


class PolicyValueHead(nn.Module):
    action_head: nn.Module
    critic_head: nn.Module

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.DistributionLike, Union[chex.Array, distrax.DistributionLike]]:

        action_distribution = self.action_head(embedding)
        value = self.critic_head(embedding)

        return action_distribution, value


class DistributionalDiscreteQNetwork(nn.Module):
    action_dim: int
    epsilon: float
    num_atoms: int
    v_min: float
    v_max: float
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        q_logits = nn.Dense(self.action_dim * self.num_atoms, kernel_init=self.kernel_init)(
            embedding
        )
        q_logits = jnp.reshape(q_logits, (-1, self.action_dim, self.num_atoms))
        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = jnp.broadcast_to(atoms, (q_values.shape[0], self.num_atoms))
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_logits, atoms


class DistributionalContinuousQNetwork(nn.Module):
    num_atoms: int
    v_min: float
    v_max: float
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        q_logits = nn.Dense(self.num_atoms, kernel_init=self.kernel_init)(embedding)
        q_dist = jax.nn.softmax(q_logits)
        q_value = jnp.sum(q_dist * atoms, axis=-1)
        # q_value = jax.lax.stop_gradient(q_value)
        atoms = jnp.broadcast_to(atoms, (*q_value.shape, self.num_atoms))
        return q_value, q_logits, atoms
