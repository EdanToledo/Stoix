from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

class Gate(nn.Module):
    output_size: int 

    @nn.compact 
    def __call__(self, x):
        x = nn.Dense(self.output_size)(x)
        x = nn.sigmoid(x)
        return x

def init_deterministic(
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 1_000 
) -> Tuple[jax.Array, jax.Array]:
    a_low = 1e-6
    a_high = 0.5 
    a = jnp.linspace(a_low, a_high, memory_size)
    b = 2 * jnp.pi / jnp.linspace(min_period, max_period, context_size)
    return a, b

class FFM(nn.Module):
    """Feedforward Memory Network."""
    trace_size: int
    context_size: int
    output_size: int

    def setup(self):
        self.pre = nn.Dense(self.trace_size)
        self.gate_in = Gate(self.trace_size)
        self.gate_out = Gate(self.output_size)
        self.skip = nn.Dense(self.output_size)
        a, b = init_deterministic(self.trace_size, self.context_size)
        self.ffa_params = (
            self.param('ffa_a', lambda rng: a),
            self.param('ffa_b', lambda rng: b)
        )
        self.mix = nn.Dense(self.output_size)
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)

    def log_gamma(self, t: jax.Array) -> jax.Array:
        a, b = self.ffa_params
        a = -jnp.abs(a).reshape((1, self.trace_size, 1))
        b = b.reshape(1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(t.shape[0], 1, 1)

    def gamma(self, t: jax.Array) -> jax.Array:
        return jnp.exp(self.log_gamma(t))

    def unwrapped_associative_update(
        self,
        carry: Tuple[jax.Array, jax.Array, jax.Array],
        incoming: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        state, i, = carry
        x, j = incoming
        state = state * self.gamma(j) + x
        return state, j + i

    def wrapped_associative_update(self, carry, incoming): 
        prev_start, state, i = carry
        start, x, j = incoming
        # Reset all elements in the carry if we are starting a new episode
        state = state * jnp.logical_not(start) 
        j = j * jnp.logical_not(start)
        incoming = x, j
        carry = (state, i)
        out = self.unwrapped_associative_update(carry, incoming)
        start_out = jnp.logical_or(start, prev_start)
        return (start_out, *out)

    def scan(
        self,
        x: jax.Array,
        state: jax.Array,
        start: jax.Array,
    ) -> jax.Array:
        """Given an input and recurrent state, this will update the recurrent state. This is equivalent
        to the inner-function g in the paper."""
        # x: [T, memory_size]
        # memory: [1, memory_size, context_size]
        T = x.shape[0]
        timestep = jnp.ones(T + 1, dtype=jnp.int32).reshape(-1, 1, 1)
        # Add context dim
        start = start.reshape(T, 1, 1)

        # Now insert previous recurrent state
        x = jnp.concatenate([state, x], axis=0)
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

        # This is not executed during inference -- method will just return x if size is 1
        _, new_state, _ = jax.lax.associative_scan(
            self.wrapped_associative_update,
            (start, x, timestep),
            axis=0,
        )
        return new_state[1:]

    def map_to_h(self, x):
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        scan_input = jnp.repeat(jnp.expand_dims(gated_x, 2), self.context_size, axis=2)
        return scan_input

    def map_from_h(self, recurrent_state, x):
        z_in = jnp.concatenate([jnp.real(recurrent_state), jnp.imag(recurrent_state)], axis=-1).reshape(
            recurrent_state.shape[0], -1
        )
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        return out

    def __call__(self, recurrent_state, inputs):
        x, resets = inputs
        z = self.map_to_h(x)
        recurrent_state = self.scan(z, recurrent_state, resets)
        out = self.map_from_h(recurrent_state, x)
        final_state = recurrent_state[-1:]
        return final_state, out

    def initialize_carry(self, batch_size: int = None):
        if batch_size is None:
            return jnp.zeros((1, self.trace_size, self.context_size), dtype=jnp.complex64)
        
        return jnp.zeros((1, batch_size, self.trace_size, self.context_size), dtype=jnp.complex64)
        


    
if __name__ == "__main__":
    m = FFM(
        output_size=4,
        trace_size=5,
        context_size=6,
    )
    s = m.initialize_carry()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    params = m.init(jax.random.PRNGKey(0), s, (x, start))
    out_state, out = m.apply(params, s, (x, start))
    
    
    BatchFFM = nn.vmap(
    FFM,
    in_axes=1, out_axes=1,
    variable_axes={'params': None},
    split_rngs={'params': False})
    
    m = BatchFFM(
        output_size=4,
        trace_size=5,
        context_size=6,
    )
    
    s = m.initialize_carry(8)
    x = jnp.ones((10, 8, 2))
    start = jnp.zeros((10, 8), dtype=bool)
    params = m.init(jax.random.PRNGKey(0), s, (x, start))
    out_state, out = m.apply(params, s, (x, start))


