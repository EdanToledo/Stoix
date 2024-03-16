from functools import partial
from typing import Any, Callable, Sequence

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from tensorflow_probability.substrates.jax.distributions import Distribution

# Different to bijectors, postprocessors simply wrap the sample and mode methods of a distribution.


class PostProcessedDistribution(Distribution):
    """A distribution that applies a postprocessing function to the samples and mode.

    This is useful for transforming the output of a distribution to a different space, such as
    rescaling the output of a tanh-transformed Normal distribution to a different range. However,
    this is not the same as a bijector, which also transforms the density function of the
    distribution. This is only useful for transforming the samples and mode of the distribution.
    For example, for an algorithm that requires taking the log probability of the samples, the
    distribution should be transformed using a bijector, not a postprocessor."""

    def __init__(
        self, distribution: Distribution, postprocessor: Callable[[chex.Array], chex.Array]
    ):
        self.distribution = distribution
        self.postprocessor = postprocessor

    def sample(self, seed: chex.PRNGKey, sample_shape: Sequence[int] = ()) -> chex.Array:
        return self.postprocessor(self.distribution.sample(seed=seed, sample_shape=sample_shape))

    def mode(self) -> chex.Array:
        return self.postprocessor(self.distribution.mode())

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.distribution, name)


def rescale_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    scale = maximum - minimum
    offset = minimum
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    return output


def clip_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    return jnp.clip(inputs, minimum, maximum)


def tanh_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    scale = maximum - minimum
    offset = minimum
    inputs = jax.nn.tanh(inputs)  # [-1, 1]
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    return output


class ScalePostProcessor(nn.Module):
    minimum: float
    maximum: float
    scale_fn: Callable[[chex.Array, float, float], chex.Array]

    @nn.compact
    def __call__(self, distribution: Distribution) -> Distribution:
        post_processor = partial(self.scale_fn, minimum=self.minimum, maximum=self.maximum)
        return PostProcessedDistribution(distribution, post_processor)


def min_max_normalize(inputs: chex.Array, epsilon: float = 1e-5) -> chex.Array:
    inputs_min = inputs.min(axis=-1, keepdims=True)
    inputs_max = inputs.max(axis=-1, keepdims=True)
    inputs_scale = inputs_max - inputs_min
    inputs_scale = jnp.where(inputs_scale < epsilon, inputs_scale + epsilon, inputs_scale)
    inputs_normed = (inputs - inputs_min) / (inputs_scale)
    return inputs_normed
