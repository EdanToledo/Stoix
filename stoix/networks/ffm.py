from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


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


class FFMCell(nn.Module):
    """The binary associative update function for the FFM."""
    trace_size: int
    context_size: int
    output_size: int

    def setup(self):
        a, b = init_deterministic(self.trace_size, self.context_size)
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

    def scan(
        self,
        state: jax.Array,
        inputs: jax.Array,
    ) -> jax.Array:
        """Execute the associative scan to update the recurrent state.
        
        Note that we do a trick here by concatenating the previou state to the inputs.
        This is allowed since the scan is associative. This ensures that the previous
        recurrent state feeds information into the scan. Without this method, we need
        separate methods for rollouts and training."""

        # Concatenate the prevous state to the inputs and scan over the result
        # This ensures the previous recurrent state contributes to the current batch
        # state: [start, (x, j)]
        # inputs: [start, (x, j)]
        scan_inputs = jax.tree.map(lambda x, s: jnp.concatenate([s, x], axis=0), inputs, state)
        new_state = jax.lax.associative_scan(        
            self.cell,
            scan_inputs,
            axis=0,
        )
        # The zeroth index corresponds to the previous recurrent state
        # We just use it to ensure continuity 
        # We do not actually want to use these values, so slice them away
        return jax.tree.map(lambda x: x[1:], new_state)

    def map_to_h(self, x):
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        scan_input = jnp.repeat(jnp.expand_dims(gated_x, 2), self.context_size, axis=2)
        return scan_input

    def map_from_h(self, recurrent_state, x):
        z_in = jnp.concatenate(
            [jnp.real(recurrent_state), jnp.imag(recurrent_state)], axis=-1
        ).reshape(recurrent_state.shape[0], -1)
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        return out

    def __call__(self, recurrent_state, inputs):
        x, resets = inputs
        z = self.map_to_h(x)
        # Relative timestep
        ts = jnp.ones(x.shape[0], dtype=jnp.int32)
        recurrent_state = self.scan(recurrent_state, ((z, ts), resets))
        # recurrent_state is ((state, timestep), reset)
        out = self.map_from_h(recurrent_state[0][0], x)
        final_state = recurrent_state[-1:]
        return final_state, out

    def initialize_carry(self, batch_size: int = None):
        return self.cell.initialize_carry(batch_size)


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
