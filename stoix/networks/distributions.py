from typing import Sequence, Union

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from distrax import MultivariateNormalDiag


class TanhMultivariateNormalDiag(MultivariateNormalDiag):
    """TanhMultivariateNormalDiag"""

    def sample(
        self, seed: Union[int, PRNGKey], sample_shape: Union[int, Sequence[int]] = ()
    ) -> Array:
        """Sample from the distribution and apply the tanh."""
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        return jnp.tanh(sample)

    def sample_unprocessed(
        self, seed: Union[int, PRNGKey], sample_shape: Union[int, Sequence[int]] = ()
    ) -> Array:
        """Sample from the distribution without applying the tanh."""
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        return sample

    def log_prob_of_unprocessed(self, value: Array) -> Array:
        """Log probability of a value in transformed distribution.
        Value is the unprocessed value. i.e. the sample before the tanh."""
        log_prob = super().log_prob(value) - jnp.sum(
            2.0 * (jnp.log(2.0) - value - jax.nn.softplus(-2.0 * value)),
            axis=-1,
        )
        return log_prob


class DeterministicDistribution(MultivariateNormalDiag):
    def sample(
        self, seed: Union[int, PRNGKey], sample_shape: Union[int, Sequence[int]] = ()
    ) -> Array:
        sample = self.loc
        return sample
