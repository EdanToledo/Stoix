from typing import Any, Optional, Sequence

import chex
import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tf_tfp
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import (
    Categorical,
    Distribution,
    MultivariateNormalDiag,
    TransformedDistribution,
)


class TanhTransformedDistribution(TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(
        self, distribution: Distribution, threshold: float = 0.999, validate_args: bool = False
    ) -> None:
        """Initialize the distribution.

        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: chex.Array) -> chex.Array:
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(event >= self._threshold, self._log_prob_right, super().log_prob(event)),
        )

    def mode(self) -> chex.Array:
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes: Any = None) -> Any:
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class DeterministicNormalDistribution(MultivariateNormalDiag):
    """Deterministic normal distribution. Always returns the mean."""

    def sample(
        self,
        seed: chex.PRNGKey = None,
        sample_shape: Sequence[int] = (),
        name: str = "sample",
    ) -> chex.Array:
        return self.loc


@tf_tfp.experimental.auto_composite_tensor
class DiscreteValuedTfpDistribution(Categorical):
    """This is a generalization of a categorical distribution.

    The support for the DiscreteValued distribution can be any real valued range,
    whereas the categorical distribution has support [0, n_categories - 1] or
    [1, n_categories]. This generalization allows us to take the mean of the
    distribution over its support.
    """

    def __init__(
        self,
        values: chex.Array,
        logits: Optional[chex.Array] = None,
        probs: Optional[chex.Array] = None,
        name: str = "DiscreteValuedDistribution",
    ):
        """Initialization.

        Args:
          values: Values making up support of the distribution. Should have a shape
            compatible with logits.
          logits: An N-D Tensor, N >= 1, representing the log probabilities of a set
            of Categorical distributions. The first N - 1 dimensions index into a
            batch of independent distributions and the last dimension indexes into
            the classes.
          probs: An N-D Tensor, N >= 1, representing the probabilities of a set of
            Categorical distributions. The first N - 1 dimensions index into a batch
            of independent distributions and the last dimension represents a vector
            of probabilities for each class. Only one of logits or probs should be
            passed in.
          name: Name of the distribution object.
        """
        parameters = dict(locals())
        self._values = np.asarray(values)

        if logits is not None:
            logits = jnp.asarray(logits)
            chex.assert_shape(logits, (..., *self._values.shape))

        if probs is not None:
            probs = jnp.asarray(probs)
            chex.assert_shape(probs, (..., *self._values.shape))

        super().__init__(logits=logits, probs=probs, name=name)

        self._parameters = parameters

    @property
    def values(self) -> chex.Array:
        return self._values

    @classmethod
    def _parameter_properties(cls, dtype: np.dtype, num_classes: Any = None) -> Any:
        return {
            "values": tfp.util.ParameterProperties(
                event_ndims=None, shape_fn=lambda shape: (num_classes,), specifies_shape=True
            ),
            "logits": tfp.util.ParameterProperties(event_ndims=1),
            "probs": tfp.util.ParameterProperties(event_ndims=1, is_preferred=False),
        }

    def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
        indices = super()._sample_n(key=key, n=n)
        return jnp.take_along_axis(self._values, indices, axis=-1)

    def mean(self) -> chex.Array:
        """Overrides the Categorical mean by incorporating category values."""
        return jnp.sum(self.probs_parameter() * self._values, axis=-1)

    def variance(self) -> chex.Array:
        """Overrides the Categorical variance by incorporating category values."""
        dist_squared = jnp.square(jnp.expand_dims(self.mean(), -1) - self._values)
        return jnp.sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self) -> chex.Array:
        return jnp.zeros((), dtype=jnp.int32)

    def _event_shape_tensor(self) -> chex.Array:
        return []
