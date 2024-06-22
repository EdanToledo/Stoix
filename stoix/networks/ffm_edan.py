from functools import partial
from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

Carry = chex.ArrayTree


class LRUCellBase(nn.Module):
    """LRU cell base class."""

    def map_to_h(self, inputs):
        """Map from the input space to the recurrent state space"""
        raise NotImplementedError

    def map_from_h(self, recurrent_state, x):
        """Map from the recurrent space to the Markov space"""
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(self, rng: chex.PRNGKey, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize the LRU cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.

        Returns:
        An initialized carry for the given RNN cell.
        """
        raise NotImplementedError

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the LRU cell."""
        raise NotImplementedError


def recurrent_associative_scan(
    cell: nn.Module,
    state: jax.Array,
    inputs: jax.Array,
    axis: int = 0,
) -> jax.Array:
    """Execute the associative scan to update the recurrent state.

    Note that we do a trick here by concatenating the previous state to the inputs.
    This is allowed since the scan is associative. This ensures that the previous
    recurrent state feeds information into the scan. Without this method, we need
    separate methods for rollouts and training."""

    # Concatenate the previous state to the inputs and scan over the result
    # This ensures the previous recurrent state contributes to the current batch
    # state: [start, (x, j)]
    # inputs: [start, (x, j)]
    scan_inputs = jax.tree.map(lambda x, s: jnp.concatenate([s, x], axis=0), inputs, state)
    new_state = jax.lax.associative_scan(
        cell,
        scan_inputs,
        axis=axis,
    )
    # The zeroth index corresponds to the previous recurrent state
    # We just use it to ensure continuity
    # We do not actually want to use these values, so slice them away
    return jax.tree.map(lambda x: x[1:], new_state)


class Gate(nn.Module):
    """Sigmoidal gating"""

    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.output_size)(x)
        x = nn.sigmoid(x)
        return x


def init_deterministic(
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 1_000
) -> Tuple[jax.Array, jax.Array]:
    """Deterministic initialization of the FFM parameters."""
    a_low = 1e-6
    a_high = 0.5
    a = jnp.linspace(a_low, a_high, memory_size)
    b = 2 * jnp.pi / jnp.linspace(min_period, max_period, context_size)
    return a, b


class FFMCell(LRUCellBase):
    """The binary associative update function for the FFM."""

    trace_size: int
    context_size: int
    output_size: int

    def setup(self):

        # Create the parameters that are explicitly used in the cells core computation
        a, b = init_deterministic(self.trace_size, self.context_size)
        self.params = (self.param("ffa_a", lambda rng: a), self.param("ffa_b", lambda rng: b))

        # Create the networks and parameters that are used when
        # mapping from input space to recurrent state space
        # This is used in the map_to_h method and is used in the
        # associative scan outer loop
        self.pre = nn.Dense(self.trace_size)
        self.gate_in = Gate(self.trace_size)
        self.gate_out = Gate(self.output_size)
        self.skip = nn.Dense(self.output_size)
        self.mix = nn.Dense(self.output_size)
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)

    def map_to_h(self, inputs):
        """Map from the input space to the recurrent state space - unlike the call function
        this explicitly expects a shape including the sequence dimension. This is used in the
        outer network that uses the associative scan."""
        x, resets = inputs
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        # We also need relative timesteps, i.e., each observation is 1 timestep newer than the previous
        ts = jnp.ones(x.shape[0], dtype=jnp.int32)
        z = jnp.repeat(jnp.expand_dims(gated_x, 2), self.context_size, axis=2)
        return (z, ts), resets

    def map_from_h(self, recurrent_state, x):
        """Map from the recurrent space to the Markov space"""
        (state, ts), reset = recurrent_state
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(
            state.shape[0], -1
        )
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        return out

    def log_gamma(self, t: jax.Array) -> jax.Array:
        a, b = self.params
        a = -jnp.abs(a).reshape((1, self.trace_size, 1))
        b = b.reshape(1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(t.shape[0], 1, 1)

    def gamma(self, t: jax.Array) -> jax.Array:
        return jnp.exp(self.log_gamma(t))

    def initialize_carry(self, batch_size: int = None):
        if batch_size is None:
            return jnp.zeros(
                (1, self.trace_size, self.context_size), dtype=jnp.complex64
            ), jnp.ones((1,), dtype=jnp.int32)

        return jnp.zeros(
            (1, batch_size, self.trace_size, self.context_size), dtype=jnp.complex64
        ), jnp.ones((1, batch_size), dtype=jnp.int32)

    def __call__(self, carry, incoming):
        (
            state,
            i,
        ) = carry
        x, j = incoming
        state = state * self.gamma(j) + x
        return state, j + i


class MemoroidResetWrapper(LRUCellBase):
    """A wrapper around memoroid cells like FFM, LRU, etc that resets
    the recurrent state upon a reset signal."""

    cell: nn.Module

    def __call__(self, carry, incoming):
        states, prev_start = carry
        xs, start = incoming

        def reset_state(start, current_state, initial_state):
            # Expand to reset all dims of state: [B, 1, 1, ...]
            expanded_start = start.reshape(-1, *([1] * (current_state.ndim - 1)))
            out = current_state * jnp.logical_not(expanded_start) + initial_state
            return out

        initial_states = self.cell.initialize_carry()
        states = jax.tree.map(partial(reset_state, start), states, initial_states)
        out = self.cell(states, xs)
        start_carry = jnp.logical_or(start, prev_start)

        return out, start_carry

    def map_to_h(self, inputs):
        return self.cell.map_to_h(inputs)

    def map_from_h(self, recurrent_state, x):
        return self.cell.map_from_h(recurrent_state, x)

    def initialize_carry(self, batch_size: int = None):
        if batch_size is None:
            # TODO: Should this be one or zero?
            return self.cell.initialize_carry(batch_size), jnp.zeros((1,), dtype=bool)

        return self.cell.initialize_carry(batch_size), jnp.zeros((1, batch_size), dtype=bool)


class ScannedLRU(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(self, recurrent_state, inputs):
        # Recurrent state should be ((state, timestep), reset)
        # Inputs should be (x, reset)
        h = self.cell.map_to_h(inputs)
        recurrent_state = recurrent_associative_scan(self.cell, recurrent_state, h)
        # recurrent_state is ((state, timestep), reset)
        out = self.cell.map_from_h(recurrent_state, x)

        # TODO: Remove this when we want to return all recurrent states instead of just the last one
        final_recurrent_state = jax.tree.map(lambda x: x[-1:], recurrent_state)
        return final_recurrent_state, out

    def initialize_carry(self, batch_size: int = None):
        return self.cell.initialize_carry(batch_size)


if __name__ == "__main__":
    m = ScannedLRU(
        cell=MemoroidResetWrapper(cell=FFMCell(output_size=4, trace_size=5, context_size=6))
    )
    s = m.initialize_carry()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    params = m.init(jax.random.PRNGKey(0), s, (x, start))
    out_state, out = m.apply(params, s, (x, start))

    print(out)
