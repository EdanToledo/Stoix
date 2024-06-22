from functools import partial
from typing import Any, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


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

    # Concatenate the prevous state to the inputs and scan over the result
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

def init_random(
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 10_000, *, key
) -> Tuple[jax.Array, jax.Array]:
    _, k1, k2 = jax.random.split(key, 3)
    a_low = 1e-6
    a_high = 0.1
    a = jax.random.uniform(k1, (memory_size,), minval=a_low, maxval=a_high)
    b = 2 * jnp.pi / jnp.exp(jax.random.uniform(k2, (context_size,), minval=jnp.log(min_period), maxval=jnp.log(max_period)))
    return a, b


class FFMCell(nn.Module):
    """The binary associative update function for the FFM."""
    trace_size: int
    context_size: int
    output_size: int
    deterministic_init: bool = True

    def setup(self):
        if self.deterministic_init: 
            a, b = init_deterministic(self.trace_size, self.context_size)
        else:
            # TODO: Will this result in the same keys for multiple FFMCells?
            key = self.make_rng("ffa_params")
            a, b = init_random(self.trace_size, self.context_size, key=key)
        self.params = (self.param("ffa_a", lambda rng: a), self.param("ffa_b", lambda rng: b))

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
            return jnp.zeros((1, self.trace_size, self.context_size), dtype=jnp.complex64), jnp.ones((1,), dtype=jnp.int32)

        return jnp.zeros((1, batch_size, self.trace_size, self.context_size), dtype=jnp.complex64), jnp.ones((1, batch_size), dtype=jnp.int32)

    def __call__(self, carry, incoming):
        (
            state,
            i,
        ) = carry
        x, j = incoming
        state = state * self.gamma(j) + x
        return state, j + i


class MemoroidResetWrapper(nn.Module):
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

    def initialize_carry(self, batch_size: int = None):
        if batch_size is None:
            # TODO: Should this be one or zero? 
            return self.cell.initialize_carry(batch_size), jnp.zeros((1,), dtype=bool) 

        return self.cell.initialize_carry(batch_size), jnp.zeros((batch_size,), dtype=bool)


class FFM(nn.Module):
    """Fast and Forgetful Memory"""

    trace_size: int
    context_size: int
    output_size: int
    cell: nn.Module

    def setup(self):
        self.pre = nn.Dense(self.trace_size)
        self.gate_in = Gate(self.trace_size)
        self.ffa = FFMCell(self.trace_size, self.context_size, self.output_size)
        self.gate_out = Gate(self.output_size)
        self.skip = nn.Dense(self.output_size)
        self.mix = nn.Dense(self.output_size)
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)

    def map_to_h(self, inputs):
        """Map from the input space to the recurrent state space"""
        x, resets = inputs
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        # We also need relative timesteps, i.e., each observation is 1 timestep newer than the previous
        ts = jnp.ones(x.shape[0], dtype=jnp.int32)
        z = jnp.repeat(jnp.expand_dims(gated_x, 2), self.context_size, axis=2)
        return (z, ts), resets

    def map_from_h(self, recurrent_state, inputs):
        """Map from the recurrent space to the Markov space"""
        (state, ts), reset = recurrent_state
        (x, start) = inputs
        z_in = jnp.concatenate(
            [jnp.real(state), jnp.imag(state)], axis=-1
        ).reshape(state.shape[0], -1)
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        return out

    def __call__(self, recurrent_state, inputs):
        # Recurrent state should be ((state, timestep), reset)
        # Inputs should be (x, reset)
        h = self.map_to_h(inputs)
        recurrent_state = recurrent_associative_scan(self.cell, recurrent_state, h)
        # recurrent_state is ((state, timestep), reset)
        out = self.map_from_h(recurrent_state, inputs)

        # TODO: Remove this when we want to return all recurrent states instead of just the last one
        final_recurrent_state = jax.tree.map(lambda x: x[-1:], recurrent_state)
        return final_recurrent_state, out

    def initialize_carry(self, batch_size: int = None):
        return self.cell.initialize_carry(batch_size)


class SFFM(nn.Module):
    """Simplified Fast and Forgetful Memory"""

    trace_size: int
    context_size: int
    hidden_size: int
    cell: nn.Module

    def setup(self):
        self.W_trace = nn.Dense(self.trace_size)
        self.W_context = Gate(self.context_size)
        self.ffa = FFMCell(self.trace_size, self.context_size, self.hidden_size, deterministic_init=False)
        self.post = nn.Sequential([
            # Default init but with smaller weights
            nn.Dense(self.hidden_size, kernel_init=nn.initializers.variance_scaling(0.01, "fan_in", "truncated_normal")),
            nn.LayerNorm(),
            nn.leaky_relu,
            nn.Dense(self.hidden_size),
            nn.LayerNorm(),
            nn.leaky_relu,
        ])

    def map_to_h(self, inputs):
        x, resets = inputs
        pre = jnp.abs(jnp.einsum("bi, bj -> bij", self.W_trace(x), self.W_context(x)))
        pre = pre / jnp.sum(pre, axis=(-2,-1), keepdims=True)
        # We also need relative timesteps, i.e., each observation is 1 timestep newer than the previous
        ts = jnp.ones(x.shape[0], dtype=jnp.int32)
        return (pre, ts), resets

    def map_from_h(self, recurrent_state, inputs):
        x, resets = inputs
        (state, ts), reset = recurrent_state
        s = state.reshape(state.shape[0], self.context_size * self.trace_size)
        eps = s.real + (s.real==0 + jnp.sign(s.real)) * 0.01
        s = s + eps
        scaled = jnp.concatenate([
            jnp.log(1 + jnp.abs(s)) * jnp.sin(jnp.angle(s)),
            jnp.log(1 + jnp.abs(s)) * jnp.cos(jnp.angle(s)),
        ], axis=-1)
        z = self.post(scaled)
        return z

    def __call__(self, recurrent_state, inputs):
        # Recurrent state should be ((state, timestep), reset)
        # Inputs should be (x, reset)
        h = self.map_to_h(inputs)
        recurrent_state = recurrent_associative_scan(self.cell, recurrent_state, h)
        # recurrent_state is ((state, timestep), reset)
        out = self.map_from_h(recurrent_state, inputs)

        # TODO: Remove this when we want to return all recurrent states instead of just the last one
        final_recurrent_state = jax.tree.map(lambda x: x[-1:], recurrent_state)
        return final_recurrent_state, out

    def initialize_carry(self, batch_size: int = None):
        return self.cell.initialize_carry(batch_size)

class StackedSFFM(nn.Module):
    """A multilayer version of SFFM"""
    cells: List[nn.Module]

    def setup(self):
        self.project = nn.Dense(cells[0].hidden_size)


    def __call__(
        self, recurrent_state: jax.Array, inputs: Any
    ) -> Tuple[jax.Array, jax.Array]:
        x, start = inputs
        x = self.project(x)
        inputs = x, start
        for i, cell in enumerate(self.cells):
            s, y = cell(recurrent_state[i], inputs)
            x = x + y
            recurrent_state[i] = s
        return y, recurrent_state 

    def initialize_carry(self, batch_size: int = None):
        return [
            c.initialize_carry(batch_size) for c in self.cells
        ]

if __name__ == "__main__":
    m = FFM(
        output_size=4,
        trace_size=5,
        context_size=6,
        cell=MemoroidResetWrapper(
            cell=FFMCell(
                output_size=4,trace_size=5,context_size=6
            )
        )
    )
    s = m.initialize_carry()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    params = m.init(jax.random.PRNGKey(0), s, (x, start))
    out_state, out = m.apply(params, s, (x, start))

    # BatchFFM = nn.vmap(
    #     FFM, in_axes=1, out_axes=1, variable_axes={"params": None}, split_rngs={"params": False}
    # )

    # m = BatchFFM(
    #     trace_size=4,
    #     context_size=5,
    #     output_size=6,
    #     cell=MemoroidResetWrapper(cell=FFMCell(4,5,6))
    # )

    # s = m.initialize_carry(8)
    # x = jnp.ones((10, 8, 2))
    # start = jnp.zeros((10, 8), dtype=bool)
    # params = m.init(jax.random.PRNGKey(0), s, (x, start))
    # out_state, out = m.apply(params, s, (x, start))

    # print(out.shape)
    # print(out_state.shape)

    # TODO: Initialize cells with different random streams so the weights are not identical
    cells = [
        SFFM(
            trace_size=4,
            context_size=5,
            hidden_size=6,
            cell=MemoroidResetWrapper(cell=FFMCell(4,5,6))
        )
        for i in range(3)
    ]
    s2fm = StackedSFFM(cells=cells)

    s = s2fm.initialize_carry()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    params = s2fm.init(jax.random.PRNGKey(0), s, (x, start))
    out_state, out = s2fm.apply(params, s, (x, start))