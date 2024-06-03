import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer using indepedent Gaussian noise
    as defined in Fortunato et al. (2018):

    y = (μ_w + σ_w * ε_w) . x + μ_b + σ_b * ε_b,

    where μ_w, μ_b, σ_w, σ_b are learnable parameters
    and ε_w, ε_b are noise random variables generated using
    Factorised Gaussian Noise.

    Attributes:
    * features (int): The number of output features.
    * sigma_zero (float): Initialization value for σ terms.
    """

    features: int
    sigma_zero: float = 0.5  # σ_0 initialization in Fortunato et al. (2017)

    def _uniform_initializer(self, key: jax.random.PRNGKey, shape: tuple) -> chex.Array:
        """
        Each element μ is sampled from independent uniform
        distributions U[−√3/p, √3/p] where p is the number of inputs.
        """
        input_dim = shape[0]  # assuming shape = (input_dim, features) or (input_dim, )
        bound = jnp.sqrt(3 / input_dim)  # √3/p
        return jax.random.uniform(key, shape, minval=-bound, maxval=bound)

    def _scale_noise(self, x: chex.Array) -> chex.Array:
        """The reference paper uses f(x) = sgn(x)√|x| as a scaling function."""
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

    def _generate_noise(self, shape: tuple) -> chex.Array:
        """Generates a Gaussian noise matrix and applies the scaling function."""
        return self._scale_noise(jax.random.normal(self.make_rng("noise"), shape))

    def _get_noise_matrix_and_vect(self, shape: tuple) -> tuple[chex.Array, chex.Array]:
        """
        Uses Factorized Gaussian Noise to generate the noise matrix ε_w
        and noise vector ε_b.
        """

        n_rows, n_cols = shape
        row_noise = self._generate_noise((n_rows,))
        col_noise = self._generate_noise((n_cols,))

        noise_matrix = jnp.outer(row_noise, col_noise)

        return noise_matrix, col_noise

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        input_dim = x.shape[-1]
        weight_matrix_shape = (input_dim, self.features)
        bias_vector_shape = (self.features,)

        sigma_init = self.sigma_zero / jnp.sqrt(input_dim)

        mu_w = self.param("mu_w", self._uniform_initializer, weight_matrix_shape)
        mu_b = self.param("mu_b", self._uniform_initializer, bias_vector_shape)

        sigma_w = self.param(
            "sigma_w",
            nn.initializers.constant(sigma_init),
            weight_matrix_shape,
        )
        sigma_b = self.param(
            "sigma_b",
            nn.initializers.constant(sigma_init),
            bias_vector_shape,
        )

        eps_w, eps_b = self._get_noise_matrix_and_vect(weight_matrix_shape)

        noisy_w = mu_w + sigma_w * eps_w
        noisy_b = mu_b + sigma_b * eps_b

        return jnp.dot(x, noisy_w) + noisy_b
