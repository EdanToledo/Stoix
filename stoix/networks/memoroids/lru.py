import functools
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from stoix.networks.memoroids.base import (
    InputEmbedding,
    Inputs,
    MemoroidCellBase,
    RecurrentState,
    Reset,
    ScanInput,
)


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def wrapped_associative_update(carry: chex.Array, incoming: chex.Array) -> Tuple[chex.Array, ...]:
    """The reset-wrapped form of the associative update.

    You might need to override this
    if you use variables in associative_update that are not from initial_state.
    This is equivalent to the h function in the paper:
    b x H -> b x H
    """
    prev_start, *carry = carry
    start, *incoming = incoming
    # Reset all elements in the carry if we are starting a new episode
    A, b = carry

    A = jnp.logical_not(start) * A + start * jnp.ones_like(A)
    b = jnp.logical_not(start) * b

    out = binary_operator_diag((A, b), incoming)
    start_out = jnp.logical_or(start, prev_start)
    return (start_out, *out)


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRUCell(MemoroidCellBase):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    hidden_state_dim: int  # hidden state dimension
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda

    def map_to_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> ScanInput:
        d_model = x.shape[-1]
        theta_log = self.param(
            "theta_log", partial(theta_init, max_phase=self.max_phase), (self.hidden_state_dim,)
        )
        nu_log = self.param(
            "nu_log", partial(nu_init, r_min=self.r_min, r_max=self.r_max), (self.hidden_state_dim,)
        )
        gamma_log = self.param("gamma_log", gamma_log_init, (nu_log, theta_log))

        B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * d_model)),
            (self.hidden_state_dim, d_model),
        )

        B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * d_model)),
            (self.hidden_state_dim, d_model),
        )

        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)

        Lambda_elements = jnp.concatenate(
            [
                jnp.ones((1, diag_lambda.shape[0])),
                Lambda_elements,
            ]
        )

        Bu_elements = jnp.concatenate(
            [
                recurrent_state,
                Bu_elements,
            ]
        )

        return (Lambda_elements, Bu_elements)

    def map_from_h(self, recurrent_states: RecurrentState, x: InputEmbedding) -> chex.Array:

        d_model = x.shape[-1]
        C_re = self.param(
            "C_re",
            partial(matrix_init, normalization=jnp.sqrt(self.hidden_state_dim)),
            (d_model, self.hidden_state_dim),
        )
        C_im = self.param(
            "C_im",
            partial(matrix_init, normalization=jnp.sqrt(self.hidden_state_dim)),
            (d_model, self.hidden_state_dim),
        )
        D = self.param("D", matrix_init, (d_model,))

        skip = x

        # Use them to compute the output of the module
        C = C_re + 1j * C_im
        x = jax.vmap(lambda h, x: (C @ h).real + D * x)(recurrent_states, x)

        x = nn.gelu(x)
        x = nn.Dense(d_model)(x) * jax.nn.sigmoid(nn.Dense(d_model)(x))  # GLU
        return skip + x  # skip connection

    def scan(
        self, start: Reset, Lambda_elements: chex.Array, Bu_elements: chex.Array
    ) -> RecurrentState:
        start = start.reshape([-1, 1])
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)
        # Compute hidden states
        _, _, xs = jax.lax.associative_scan(
            wrapped_associative_update, (start, Lambda_elements, Bu_elements)
        )
        return xs[1:]

    @functools.partial(
        nn.vmap,
        variable_axes={"params": None},
        in_axes=(0, 1),
        out_axes=(0, 1),
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, recurrent_state: RecurrentState, inputs: Inputs
    ) -> Tuple[RecurrentState, chex.Array]:
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""

        # Add a sequence dimension to the recurrent state
        recurrent_state = jnp.expand_dims(recurrent_state, 0)

        x, starts = inputs

        (Lambda_elements, Bu_elements) = self.map_to_h(recurrent_state, x)

        # Compute hidden states
        hidden_states = self.scan(starts, Lambda_elements, Bu_elements)

        outputs = self.map_from_h(hidden_states, x)

        # Already has sequence dim removed
        new_hidden_state = hidden_states[-1]

        return new_hidden_state, outputs

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> RecurrentState:
        return jnp.zeros((batch_size, self.hidden_state_dim), dtype=jnp.complex64)
