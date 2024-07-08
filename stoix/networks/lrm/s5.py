# flake8: noqa
import functools
from typing import Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import Initializer
from jax import random
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh

from stoix.networks.lrm.base import (
    InputEmbedding,
    Inputs,
    LRMCellBase,
    RecurrentState,
    Reset,
    ScanInput,
)

# S5 code taken and modified from https://github.com/luchris429/popjaxrl


def log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1) -> Initializer:
    """Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.
    Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
    Returns:
        init function
    """

    def init(key: chex.PRNGKey, shape: Sequence[int]) -> chex.Array:
        """Init function
        Args:
            key: jax random key
            shape tuple: desired shape
        Returns:
            sampled log_step (float32)
        """
        return random.uniform(key, shape) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)

    return init


def init_log_steps(key: chex.PRNGKey, input: Tuple[int, float, float]) -> chex.Array:
    """Initialize an array of learnable timescale parameters
    Args:
        key: jax random key
        input: tuple containing the array shape H and
               dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
    """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return jnp.array(log_steps)


def init_VinvB(
    init_fun: Initializer, rng: chex.PRNGKey, shape: Sequence[int], Vinv: chex.Array
) -> chex.Array:
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax random key to be used with init function.
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H,2)
    """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return jnp.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key: chex.PRNGKey, shape: Sequence[int]) -> chex.Array:
    """Sample C with a truncated normal distribution with standard deviation 1.
    Args:
        key: jax random key
        shape (tuple): desired shape, of length 3, (H,P,_)
    Returns:
        sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


def init_CV(
    init_fun: Initializer, rng: chex.PRNGKey, shape: Sequence[int], V: chex.Array
) -> chex.Array:
    """Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax random key to be used with init function.
         shape (tuple): desired shape  (H,P)
         V: (complex64)     the eigenvectors used for initialization
     Returns:
         C_tilde (complex64) of shape (H,P,2)
    """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return jnp.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


# Discretization functions
def discretize_bilinear(
    Lambda: chex.Array, B_tilde: chex.Array, Delta: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(
    Lambda: chex.Array, B_tilde: chex.Array, Delta: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator_reset(
    q_i: chex.Array, q_j: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i) * (1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )


def make_HiPPO(N: int) -> chex.Array:
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N: int) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N: int) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


class S5Cell(LRMCellBase):
    d_model: int
    state_size: int
    blocks: int = 1

    activation: str = "gelu"
    do_norm: bool = True
    prenorm: bool = True
    do_gtrxl_norm: bool = True

    C_init: str = "lecun_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    def setup(self) -> None:
        """Initializes parameters once and performs discretization each time
        the SSM is applied to a sequence
        """
        self.ssm_size = self.state_size * 2

        block_size = int(self.ssm_size / self.blocks)
        Lambda, _, _, V, _ = make_DPLR_HiPPO(self.ssm_size)
        block_size = block_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        self.H = self.d_model
        self.P = self.state_size
        self.Lambda_re_init = Lambda.real
        self.Lambda_im_init = Lambda.imag
        self.V = V
        self.Vinv = Vinv

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param(
            "B", lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv), B_shape
        )
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError("C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param(
                    "C1", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape
                )
                self.C2 = self.param(
                    "C2", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape
                )

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = jnp.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param(
                    "C", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape
                )

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                "Discretization method {} not implemented".format(self.discretization)
            )

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        self.norm = nn.LayerNorm()

    def map_to_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> ScanInput:

        if self.prenorm and self.do_norm:
            x = self.norm(x)

        Lambda_elements = self.Lambda_bar * jnp.ones((x.shape[0], self.Lambda_bar.shape[0]))
        Bu_elements = jax.vmap(lambda u: self.B_bar @ u)(x)

        Lambda_elements = jnp.concatenate(
            [
                jnp.ones((1, self.Lambda_bar.shape[0])),
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

    def scan(
        self, resets: Reset, Lambda_elements: chex.Array, Bu_elements: chex.Array
    ) -> RecurrentState:

        resets = jnp.concatenate(
            [
                jnp.zeros(1),
                resets,
            ]
        )
        _, xs, _ = jax.lax.associative_scan(
            binary_operator_reset, (Lambda_elements, Bu_elements, resets)
        )
        xs = xs[1:]

        return xs

    def map_from_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> chex.Array:
        skip = x
        if self.conj_sym:
            x = jax.vmap(lambda x: 2 * (self.C_tilde @ x).real)(recurrent_state)
        else:
            x = jax.vmap(lambda x: (self.C_tilde @ x).real)(recurrent_state)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(x)
        x = x + Du

        if self.do_gtrxl_norm:
            x = self.norm(x)

        if self.activation in ["full_glu"]:
            x = nn.gelu(x)
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        elif self.activation in ["half_glu1"]:
            x = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x))
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            x = nn.gelu(x)

        x = skip + x
        if not self.prenorm and self.do_norm:
            x = self.norm(x)

        return x

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
        return jnp.zeros((batch_size, self.state_size), dtype=jnp.complex64)
