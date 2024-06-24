from functools import partial
from typing import Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

# Typing aliases
Carry = chex.ArrayTree

HiddenState = chex.Array
Timestep = chex.Array
Reset = chex.Array

RecurrentState = Tuple[HiddenState, Timestep]

InputEmbedding = chex.Array
Inputs = Tuple[InputEmbedding, Reset]


def debug_shape(x):
    return jax.tree.map(lambda x: x.shape, x)


class MemoroidCellBase(nn.Module):
    """Memoroid cell base class."""

    def map_to_h(self, inputs: Inputs) -> RecurrentState:
        """Map from the input space to the recurrent state space"""
        raise NotImplementedError

    def map_from_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> HiddenState:
        """Map from the recurrent space to the Markov space"""
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(
        self, batch_size: Optional[int] = None, rng: Optional[chex.PRNGKey] = None
    ) -> RecurrentState:
        """Initialize the Memoroid cell carry.

        Args:
            batch_size: the batch size of the carry.
            rng: random number generator passed to the init_fn.

        Returns:
        An initialized carry for the given RNN cell.
        """
        raise NotImplementedError

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the cell."""
        raise NotImplementedError


def recurrent_associative_scan(
    cell: nn.Module,
    state: RecurrentState,
    inputs: RecurrentState,
    axis: int = 0,
) -> RecurrentState:
    """Execute the associative scan to update the recurrent state.

    Note that we do a trick here by concatenating the previous state to the inputs.
    This is allowed since the scan is associative. This ensures that the previous
    recurrent state feeds information into the scan. Without this method, we need
    separate methods for rollouts and training."""

    # Concatenate the previous state to the inputs and scan over the result
    # This ensures the previous recurrent state contributes to the current batch

    # We need to add a dummy start signal to the inputs
    dummy_start = jnp.zeros(inputs[-1].shape[1:], dtype=bool)[jnp.newaxis, ...]
    # Add it to the state i.e. (state, timestep) -> ((state, time), reset)
    state = (state, dummy_start)
    scan_inputs = jax.tree.map(lambda s, x: jnp.concatenate([s, x], axis=axis), state, inputs)
    new_state = jax.lax.associative_scan(
        cell,
        scan_inputs,
        axis=axis,
    )

    # Get rid of the reset signal i.e. ((state, time), reset) -> (state, time)
    new_state, _ = new_state

    # The zeroth index corresponds to the previous recurrent state
    # We just use it to ensure continuity
    # We do not actually want to use these values, so slice them away
    return jax.tree.map(
        lambda x: jax.lax.slice_in_dim(x, start_index=1, limit_index=None, axis=axis), new_state
    )


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
) -> Tuple[chex.Array, chex.Array]:
    """Deterministic initialization of the FFM parameters."""
    a_low = 1e-6
    a_high = 0.5
    a = jnp.linspace(a_low, a_high, memory_size)
    b = 2 * jnp.pi / jnp.linspace(min_period, max_period, context_size)
    return a, b


class FFMCell(MemoroidCellBase):
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

    def map_to_h(self, x: InputEmbedding) -> RecurrentState:
        """Map from the input space to the recurrent state space - unlike the call function
        this explicitly expects a shape including the sequence dimension. This is used in the
        outer network that uses the associative scan."""
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        # We also need relative timesteps, i.e., each observation is 1 timestep newer than the previous
        ts = jnp.ones(x.shape[0:2], dtype=jnp.int32)
        z = jnp.repeat(jnp.expand_dims(gated_x, 3), self.context_size, axis=3)
        return (z, ts)

    def map_from_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> HiddenState:
        """Map from the recurrent space to the Markov space"""
        state, _ = recurrent_state
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(
            state.shape[0], state.shape[1], -1
        )
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        return out

    def log_gamma(self, t: chex.Array) -> chex.Array:
        a, b = self.params
        a = -jnp.abs(a).reshape((1, 1, self.trace_size, 1))
        b = b.reshape(1, 1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(t.shape[0], t.shape[1], 1, 1)

    def gamma(self, t: chex.Array) -> chex.Array:
        return jnp.exp(self.log_gamma(t))

    @nn.nowrap
    def initialize_carry(
        self, batch_size: Optional[int] = None, rng: Optional[chex.PRNGKey] = None
    ) -> RecurrentState:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        carry_shape = (1, self.trace_size, self.context_size)
        t_shape = (1,)
        if batch_size is not None:
            carry_shape = (carry_shape[0], batch_size, *carry_shape[1:])
            t_shape = (*t_shape, batch_size)
        return jnp.zeros(carry_shape, dtype=jnp.complex64), jnp.ones(t_shape, dtype=jnp.int32)

    def __call__(self, carry: RecurrentState, incoming):
        (
            state,
            i,
        ) = carry
        x, j = incoming
        state = state * self.gamma(j) + x
        return state, j + i


class MemoroidResetWrapper(MemoroidCellBase):
    """A wrapper around memoroid cells like FFM, LRU, etc that resets
    the recurrent state upon a reset signal."""

    cell: nn.Module

    def __call__(self, carry, incoming, rng=None):
        states, prev_start = carry
        xs, start = incoming

        def reset_state(start: Reset, current_state, initial_state):
            # Expand to reset all dims of state: [1, B, 1, ...]
            assert initial_state.ndim == current_state.ndim
            expanded_start = start.reshape(-1, start.shape[1], *([1] * (current_state.ndim - 2)))
            out = current_state * jnp.logical_not(expanded_start) + initial_state
            return out

        # Add an extra dim, as start will be [Batch] while intialize carry expects [Batch, Feature]
        initial_states = self.cell.initialize_carry(rng=rng, batch_size=start.shape[1])
        states = jax.tree.map(partial(reset_state, start), states, initial_states)
        out = self.cell(states, xs)
        start_carry = jnp.logical_or(start, prev_start)

        return out, start_carry

    def map_to_h(self, x: InputEmbedding) -> RecurrentState:
        return self.cell.map_to_h(x)

    def map_from_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> HiddenState:
        return self.cell.map_from_h(recurrent_state, x)

    @nn.nowrap
    def initialize_carry(
        self, batch_size: Optional[int] = None, rng: Optional[chex.PRNGKey] = None
    ) -> RecurrentState:
        return self.cell.initialize_carry(batch_size, rng)


class ScannedMemoroid(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(
        self, recurrent_state: RecurrentState, inputs: Inputs
    ) -> Tuple[RecurrentState, HiddenState]:
        """Apply the ScannedMemoroid.
        This takes in a sequence of batched states and inputs.
        The recurrent state that is used requires no sequence dimension but does require a batch dimension."""
        # Recurrent state should be (state, timestep)
        # Inputs should be (x, reset)

        # Unsqueeze the recurrent state to add the sequence dimension of size 1
        recurrent_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), recurrent_state)

        x, resets = inputs
        h = self.cell.map_to_h(x)
        recurrent_state = recurrent_associative_scan(self.cell, recurrent_state, (h, resets))
        # recurrent_state is (state, timestep)
        out = self.cell.map_from_h(recurrent_state, x)

        # TODO: Remove this when we want to return all recurrent states instead of just the last one
        final_recurrent_state = jax.tree.map(lambda x: x[-1:], recurrent_state)

        # Squeeze the sequence dimension of 1 out
        final_recurrent_state = jax.tree.map(lambda x: jnp.squeeze(x, 0), final_recurrent_state)

        return final_recurrent_state, out

    @nn.nowrap
    def initialize_carry(
        self, batch_size: Optional[int] = None, rng: Optional[chex.PRNGKey] = None
    ) -> RecurrentState:
        """Initialize the carry for the ScannedMemoroid. This returns the carry in the shape [Batch, ...] i.e. it contains no sequence dimension"""
        # We squeeze the sequence dim of 1 out.
        return jax.tree.map(lambda x: x.squeeze(0), self.cell.initialize_carry(batch_size, rng))


if __name__ == "__main__":
    BatchFFM = ScannedMemoroid

    m = BatchFFM(
        cell=MemoroidResetWrapper(cell=FFMCell(output_size=4, trace_size=5, context_size=6))
    )

    batch_size = 8
    time_steps = 10

    y = jnp.ones((time_steps, batch_size, 2))
    s = m.initialize_carry(batch_size)
    start = jnp.zeros((time_steps, batch_size), dtype=bool)
    params = m.init(jax.random.PRNGKey(0), s, (y, start))
    out_state, out = m.apply(params, s, (y, start))

    out = jnp.swapaxes(out, 0, 1)

    print(out)
    print(debug_shape(out_state))
