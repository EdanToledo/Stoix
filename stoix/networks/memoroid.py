from functools import partial
from typing import Optional, Tuple

import chex
import flax.linen as nn
import flax
import jax
import optax
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

    scan_inputs = jax.tree.map(lambda s, x: jnp.concatenate([s, x], axis=axis), state, inputs)
    new_state = jax.lax.associative_scan(
        cell,
        scan_inputs,
        axis=axis,
    )

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
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 1024
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
        (state, _), _ = recurrent_state
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
        
        return jnp.zeros(carry_shape, dtype=jnp.complex64), jnp.zeros(t_shape, dtype=jnp.int32)

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
        states, prev_carry_reset_flag = carry
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
        carry_reset_flag = jnp.logical_or(start, prev_carry_reset_flag)

        return out, carry_reset_flag

    def map_to_h(self, x: InputEmbedding) -> RecurrentState:
        return self.cell.map_to_h(x)

    def map_from_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> HiddenState:
        return self.cell.map_from_h(recurrent_state, x)

    @nn.nowrap
    def initialize_carry(
        self, batch_size: Optional[int] = None, rng: Optional[chex.PRNGKey] = None
    ) -> RecurrentState:
        return self.cell.initialize_carry(batch_size, rng), jnp.zeros((1, batch_size), dtype=bool)


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
        # TODO: In the original implementation, the recurrent timestep is also one
        # recurrent_state = (
        #     (recurrent_state[0][0],
        #     jnp.ones_like(recurrent_state[0][1])),
        #     recurrent_state[1]
        # )
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


def test_reset_wrapper():
    """Validate that the reset wrapper works as expected"""
    BatchFFM = ScannedMemoroid

    m = BatchFFM(
        cell=MemoroidResetWrapper(cell=FFMCell(output_size=2, trace_size=2, context_size=3))
    )

    batch_size = 4 
    time_steps = 100
    # Have a batched version with one episode per batch
    # and collapse it into a single episode with a single batch (but same start/resets)
    # results should be identical
    batched_starts = jnp.ones([batch_size], dtype=bool)
    # batched_starts = jnp.concatenate([
    #     jnp.zeros([time_steps // 2, batch_size], dtype=bool),
    #     batched_starts.reshape(1, -1),
    #     jnp.zeros([time_steps // 2 - 1, batch_size], dtype=bool)
    # ], axis=0)
    batched_starts = jax.random.uniform(jax.random.PRNGKey(0), (time_steps, batch_size)) < 0.1
    contig_starts = jnp.swapaxes(batched_starts, 1, 0).reshape(-1, 1)

    x_batched = jnp.arange(time_steps * batch_size * 2).reshape((time_steps, batch_size, 2))
    x_contig = jnp.swapaxes(x_batched, 1, 0).reshape(-1, 1, 2)
    batched_s = m.initialize_carry(batch_size)
    contig_s = m.initialize_carry(1)
    params = m.init(jax.random.PRNGKey(0), batched_s, (x_batched, batched_starts))


    ((batched_out_state, batched_ts), batched_reset), batched_out = m.apply(params, batched_s, (x_batched, batched_starts))
    ((contig_out_state, contig_ts), contig_reset), contig_out = m.apply(params, contig_s, (x_contig, contig_starts))

    # This should be nearly zero (1e-10 or something)
    state_error = jnp.linalg.norm(contig_out_state - batched_out_state[-1], axis=-1).sum()
    print("state error", state_error)
    out_error = jnp.linalg.norm(batched_out - jnp.swapaxes(contig_out.reshape(batch_size, time_steps, -1), 1, 0), axis=-1).sum()
    print("out error", out_error)
    print(batched_ts, contig_ts)

def test_reset_wrapper_ts():
    BatchFFM = ScannedMemoroid

    m = BatchFFM(
        cell=MemoroidResetWrapper(cell=FFMCell(output_size=2, trace_size=2, context_size=3))
    )

    batch_size = 2 
    time_steps = 10
    # Have a batched version with one episode per batch
    # and collapse it into a single episode with a single batch (but same start/resets)
    # results should be identical
    batched_starts = jnp.array([
        [False, False, True, False, False, True, True, False, False, False],
        [False, False, True, False, False, True, True, False, False, False],
    ]).T

    x_batched = jnp.arange(time_steps * batch_size * 2).reshape((time_steps, batch_size, 2)).astype(jnp.float32)
    batched_s = m.initialize_carry(batch_size)
    params = m.init(jax.random.PRNGKey(0), batched_s, (x_batched, batched_starts))


    ((batched_out_state, batched_ts), batched_reset), batched_out = m.apply(params, batched_s, (x_batched, batched_starts))
    print(batched_ts == 4)



def train_memorize():
    BatchFFM = ScannedMemoroid

    m = BatchFFM(
        cell=MemoroidResetWrapper(cell=FFMCell(output_size=128, trace_size=32, context_size=4))
    )

    batch_size = 1
    rem_ts = 10
    time_steps = rem_ts * 5
    obs_space = 2
    rng = jax.random.PRNGKey(0)
    x = jax.random.randint(rng, (time_steps, batch_size), 0, obs_space).reshape(-1, 1, 1)
    y = jnp.repeat(x[::rem_ts], x.shape[0] // x[::rem_ts].shape[0]).reshape(-1, 1)
    start = jnp.zeros([time_steps, batch_size], dtype=bool).at[::rem_ts].set(True)
    #start = jnp.zeros([time_steps, batch_size], dtype=bool)
    #start = jnp.ones([time_steps, batch_size], dtype=bool)

    s = m.initialize_carry(batch_size)
    params = m.init(jax.random.PRNGKey(0), s, (x, start))

    def error(params, x, start, key):
        s = m.initialize_carry(batch_size)
        x = jax.random.randint(key, (time_steps, batch_size), 0, obs_space).reshape(-1, 1, 1)
        out_state, y_hat = m.apply(params, s, (x, start))
        return jnp.mean((y - y_hat) ** 2)

    optimizer = optax.adam(learning_rate=0.002)
    state = optimizer.init(params)
    loss_fn = jax.jit(jax.value_and_grad(error))
    for step in range(10_000):
        rng = jax.random.split(rng)[0]
        loss, grads = loss_fn(params, x, start, rng)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        print(f"Step {step+1}, Loss: {loss}")


if __name__ == "__main__":
    # BatchFFM = ScannedMemoroid

    # m = BatchFFM(
    #     cell=MemoroidResetWrapper(cell=FFMCell(output_size=4, trace_size=5, context_size=6))
    # )

    # batch_size = 8
    # time_steps = 10

    # y = jnp.ones((time_steps, batch_size, 2))
    # s = m.initialize_carry(batch_size)
    # start = jnp.zeros((time_steps, batch_size), dtype=bool)
    # params = m.init(jax.random.PRNGKey(0), s, (y, start))
    # out_state, out = m.apply(params, s, (y, start))

    # out = jnp.swapaxes(out, 0, 1)

    # print(out)
    # print(debug_shape(out_state))

    #test_reset_wrapper()
    #test_reset_wrapper_ts()
    train_memorize()
