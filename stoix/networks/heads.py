from typing import Optional, Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, lecun_normal, orthogonal
from tensorflow_probability.substrates.jax.distributions import (
    Categorical,
    Deterministic,
    Independent,
    MultivariateNormalDiag,
    Normal,
)

from stoix.networks.distributions import (
    DiscreteValuedTfpDistribution,
    TanhTransformedDistribution,
)


class CategoricalHead(nn.Module):
    action_dim: Union[int, Sequence[int]]
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Categorical:

        logits = nn.Dense(np.prod(self.action_dim), kernel_init=self.kernel_init)(embedding)

        if not isinstance(self.action_dim, int):
            logits = logits.reshape(self.action_dim)

        return Categorical(logits=logits)


class NormalTanhDistributionHead(nn.Module):

    action_dim: int
    min_scale: float = 1e-3
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Independent:

        loc = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        scale = jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)) + self.min_scale
        distribution = Normal(loc=loc, scale=scale)

        return Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)


class MultivariateNormalDiagHead(nn.Module):

    action_dim: int
    init_scale: float = 0.3
    min_scale: float = 1e-6
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.DistributionLike:
        loc = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        scale = jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding))
        scale *= self.init_scale / jax.nn.softplus(0.0)
        scale += self.min_scale
        return MultivariateNormalDiag(loc=loc, scale_diag=scale)


class DeterministicHead(nn.Module):
    action_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:

        x = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)

        return Deterministic(x)


class ScalarCriticHead(nn.Module):
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return nn.Dense(1, kernel_init=self.kernel_init)(embedding).squeeze(axis=-1)


class CategoricalCriticHead(nn.Module):

    num_bins: int = 601
    vmax: Optional[float] = None
    vmin: Optional[float] = None
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.DistributionLike:
        vmax = self.vmax if self.vmax is not None else 0.5 * (self.num_bins - 1)
        vmin = self.vmin if self.vmin is not None else -1.0 * vmax

        output = DiscreteValuedTfpHead(
            vmin=vmin,
            vmax=vmax,
            logits_shape=(1,),
            num_atoms=self.num_bins,
            kernel_init=self.kernel_init,
        )(embedding)

        return output


class DiscreteValuedTfpHead(nn.Module):
    """Represents a parameterized discrete valued distribution.

    The returned distribution is essentially a `tfd.Categorical` that knows its
    support and thus can compute the mean value.
    If vmin and vmax have shape S, this will store the category values as a
    Tensor of shape (S*, num_atoms).

    Args:
        vmin: Minimum of the value range
        vmax: Maximum of the value range
        num_atoms: The atom values associated with each bin.
        logits_shape: The shape of the logits, excluding batch and num_atoms
        dimensions.
        kernel_init: The initializer for the dense layer.
    """

    vmin: float
    vmax: float
    num_atoms: int
    logits_shape: Optional[Sequence[int]] = None
    kernel_init: Initializer = lecun_normal()

    def setup(self) -> None:
        self._values = np.linspace(self.vmin, self.vmax, num=self.num_atoms, axis=-1)
        if not self.logits_shape:
            logits_shape = ()
        self._logits_shape = logits_shape + (self.num_atoms,)
        self._logits_size = np.prod(self._logits_shape)

    def __call__(self, inputs: chex.Array) -> distrax.DistributionLike:
        net = nn.Dense(self._logits_size, kernel_init=self.kernel_init)
        logits = net(inputs)
        logits = logits.reshape(logits.shape[:-1] + self._logits_shape)
        return DiscreteValuedTfpDistribution(values=self._values, logits=logits)


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
    def __call__(self, embedding: chex.Array) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        q_logits = nn.Dense(self.action_dim * self.num_atoms, kernel_init=self.kernel_init)(embedding)
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
    def __call__(self, embedding: chex.Array) -> Tuple[distrax.EpsilonGreedy, chex.Array, chex.Array]:
        atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)
        q_logits = nn.Dense(self.num_atoms, kernel_init=self.kernel_init)(embedding)
        q_dist = jax.nn.softmax(q_logits)
        q_value = jnp.sum(q_dist * atoms, axis=-1)
        atoms = jnp.broadcast_to(atoms, (*q_value.shape, self.num_atoms))
        return q_value, q_logits, atoms


class QuantileDiscreteQNetwork(nn.Module):
    action_dim: int
    epsilon: float
    num_quantiles: int
    kernel_init: Initializer = lecun_normal()

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Tuple[distrax.EpsilonGreedy, chex.Array]:
        q_logits = nn.Dense(self.action_dim * self.num_quantiles, kernel_init=self.kernel_init)(embedding)
        q_dist = jnp.reshape(q_logits, (-1, self.action_dim, self.num_quantiles))
        q_values = jnp.mean(q_dist, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_dist
