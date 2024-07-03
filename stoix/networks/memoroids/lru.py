from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import Initializer

from stoix.networks.memoroids.base import (
    InputEmbedding,
    Inputs,
    MemoroidCellBase,
    RecurrentState,
    Reset,
    ScanInput,
)

# NOT WORKING YET

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


def matrix_init(normalization: float = 1.0) -> Initializer:
    def init(
        key: chex.PRNGKey, shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization

    return init


def nu_init(r_min: float, r_max: float) -> Initializer:
    def init(
        key: chex.PRNGKey, shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
        return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))

    return init


def theta_init(max_phase: float) -> Initializer:
    def init(
        key: chex.PRNGKey, shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        u = jax.random.uniform(key, shape=shape, dtype=dtype)
        return jnp.log(max_phase * u)

    return init


def gamma_log_init(
    lamb: Tuple[Union[float, jnp.ndarray], Union[float, jnp.ndarray]]
) -> Initializer:
    def init(
        key: chex.PRNGKey, shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        nu, theta = lamb
        diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
        return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

    return init


class LRUCell(MemoroidCellBase):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_model: int  # input and output dimensions
    d_hidden: int  # hidden state dimension
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda

    def setup(self):

        self.theta_log = self.param("theta_log", theta_init(self.max_phase), (self.d_hidden,))
        self.nu_log = self.param("nu_log", nu_init(self.r_min, self.r_max), (self.d_hidden,))
        self.gamma_log = self.param(
            "gamma_log", gamma_log_init((self.nu_log, self.theta_log)), (self.d_hidden,)
        )

        self.B_re = self.param(
            "B_re",
            matrix_init(normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            matrix_init(normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C_re = self.param(
            "C_re",
            matrix_init(normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.C_im = self.param(
            "C_im",
            matrix_init(normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init(normalization=1), (self.d_model,))

        self.normalization = nn.LayerNorm()
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)

    def map_to_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> ScanInput:
        x = self.normalization(x)
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x.astype(jnp.complex64))

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

        skip = x

        C = self.C_re + 1j * self.C_im

        # Use them to compute the output of the module
        x = jax.vmap(lambda x, u: (C @ x).real + self.D * u)(recurrent_states, x)

        x = jax.nn.gelu(x)
        o1 = self.out1(x)
        x = o1 * jax.nn.sigmoid(self.out2(x))  # GLU
        return skip + x  # skip connection

    def scan(self, start, Lambda_elements, Bu_elements) -> RecurrentState:

        # Compute hidden states
        _, _, xs = jax.lax.associative_scan(
            wrapped_associative_update, (start, Lambda_elements, Bu_elements)
        )

        return xs[1:]

    def __call__(self, recurrent_state: RecurrentState, inputs: Inputs):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""

        x, start = inputs

        (Lambda_elements, Bu_elements) = self.map_to_h(recurrent_state, x)

        start = start.reshape([-1, 1])
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

        new_recurrent_states = self.scan(start, Lambda_elements, Bu_elements)

        outputs = self.map_from_h(new_recurrent_states, x)

        return new_recurrent_states[None, -1], outputs

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> RecurrentState:
        return jnp.zeros((1, self.d_hidden), dtype=jnp.complex64)


if __name__ == "__main__":
    LRUModel = LRUCell(d_model=2, d_hidden=4)

    m = LRUModel

    batch_size = 1
    time_steps = 10

    y = jnp.ones((time_steps, 2))
    s = m.initialize_carry(batch_size)
    start = jnp.zeros((time_steps,), dtype=bool)
    params = m.init(jax.random.PRNGKey(0), s, (y, start))
    out_state, out = m.apply(params, s, (y, start))

    print(out)
