import jax
import jax.numpy as jnp
from flax import linen as nn


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer using indepedent Gaussian noise
    as defined in Fortunato et al. (2018):

    y = (μ_w + σ_w * ε_w) . x + μ_b + σ_b * ε_b,

    where μ_w, μ_b, σ_w, σ_b are learnable parameters
    and ε_w, ε_b are noise random variables.
    """

    features: int
    sigma_init: float = 0.017  # σ initialization in Fortunato et al. (2017)

    def _uniform_init(self, key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
        """
        Each element μ is sampled from independent uniform
        distributions U[−√3/p, √3/p] where p is the number of inputs.
        """
        input_dim = shape[0]  # assuming shape = (input_dim, features)
        bound = jnp.sqrt(3 / input_dim)
        return jax.random.uniform(
            key,
            shape,
            minval=-bound,
            maxval=bound,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_dim = x.shape[-1]

        mu_w = self.param("mu_w", self._uniform_init, (input_dim, self.features))
        mu_b = self.param("mu_b", self._uniform_init, (self.features))

        sigma_w = self.param(
            "sigma_w",
            nn.initializers.constant(self.sigma_init),
            (input_dim, self.features),
        )
        sigma_b = self.param(
            "sigma_b",
            nn.initializers.constant(self.sigma_init),
            (self.features),
        )

        eps_w = jax.random.normal(self.make_rng("params"), (input_dim, self.features))
        eps_b = jax.random.normal(self.make_rng("params"), (self.features))

        noisy_w = mu_w + sigma_w * eps_w
        noisy_b = mu_b + sigma_b * eps_b

        return jnp.dot(noisy_w, x) + noisy_b
