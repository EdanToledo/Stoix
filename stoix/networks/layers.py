from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, Initializer, PrecisionLike

from stoix.networks.utils import parse_activation_fn

default_kernel_init = initializers.lecun_normal()


class StackedRNN(nn.Module):
    """
    A class representing a stacked recurrent neural network (RNN).

    Attributes:
        rnn_size (int): The size of the hidden state for each RNN cell.
        rnn_cls (nn.Module): The class for the RNN cell to be used.
        num_layers (int): The number of RNN layers.
        activation_fn (str): The activation function to use in each RNN cell (default is "tanh").
    """

    rnn_size: int
    rnn_cls: nn.Module
    num_layers: int
    activation_fn: str = "sigmoid"

    def setup(self) -> None:
        """Set up the RNN cells for the stacked RNN."""
        self.cells = [
            self.rnn_cls(
                features=self.rnn_size, activation_fn=parse_activation_fn(self.activation_fn)
            )
            for _ in range(self.num_layers)
        ]

    def __call__(
        self, all_rnn_states: List[chex.ArrayTree], x: chex.Array
    ) -> Tuple[List[chex.ArrayTree], chex.Array]:
        """
        Run the stacked RNN cells on the input.

        Args:
            all_rnn_states (List[chex.ArrayTree]): List of RNN states for each layer.
            x (chex.Array): Input to the RNN.

        Returns:
            Tuple[List[chex.ArrayTree], chex.Array]: A tuple containing the a list of
                the RNN states of each RNN and the output of the last layer.
        """
        # Ensure all_rnn_states is a list
        if not isinstance(all_rnn_states, list):
            all_rnn_states = [all_rnn_states]

        assert (
            len(all_rnn_states) == self.num_layers
        ), f"Expected {self.num_layers} RNN states, but got {len(all_rnn_states)}."

        new_states = []
        for cell, rnn_state in zip(self.cells, all_rnn_states):
            new_rnn_state, x = cell(rnn_state, x)
            new_states.append(new_rnn_state)

        return new_states, x


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer using independent Gaussian noise
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
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    sigma_zero: float = 0.5  # σ_0 initialization in Fortunato et al. (2017)

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
    def __call__(self, inputs: chex.Array) -> chex.Array:

        input_dim = jnp.shape(inputs)[-1]
        kernel_shape = (input_dim, self.features)
        bias_vector_shape = (self.features,)
        sigma_init = self.sigma_zero / jnp.sqrt(input_dim)

        kernel = self.param(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.param_dtype,
        )

        sigma_w = self.param(
            "sigma_w",
            nn.initializers.constant(sigma_init),
            kernel_shape,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, bias_vector_shape, self.param_dtype)
            sigma_b = self.param(
                "sigma_b",
                nn.initializers.constant(sigma_init),
                bias_vector_shape,
            )
        else:
            bias = None
            sigma_b = None

        inputs, kernel, bias, sigma_w, sigma_b = promote_dtype(
            inputs, kernel, bias, sigma_w, sigma_b, dtype=self.dtype
        )

        eps_w, eps_b = self._get_noise_matrix_and_vect(kernel_shape)

        noisy_kernel = kernel + sigma_w * eps_w

        y = jax.lax.dot_general(
            inputs,
            noisy_kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            noisy_bias = bias + sigma_b * eps_b
            y += jnp.reshape(noisy_bias, (1,) * (y.ndim - 1) + (-1,))

        return y
