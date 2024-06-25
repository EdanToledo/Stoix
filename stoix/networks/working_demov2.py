from typing import Tuple

import chex
import jax
import optax
from flax import linen as nn
from jax import numpy as jnp

RecurrentState = chex.Array
Reset = chex.Array
Timestep = chex.Array
InputEmbedding = chex.Array
Inputs = Tuple[InputEmbedding, Reset]
ScanInput = chex.Array


def init_deterministic(
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 1_000
) -> Tuple[chex.Array, chex.Array]:
    a_low = 1e-6
    a_high = 0.5
    a = jnp.linspace(a_low, a_high, memory_size)
    b = 2 * jnp.pi / jnp.linspace(min_period, max_period, context_size)
    return a, b


class Gate(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return jax.nn.sigmoid(nn.Dense(self.output_size)(x))


class FFM(nn.Module):
    trace_size: int
    context_size: int
    output_size: int

    def setup(self) -> None:

        # Create the FFM parameters
        a, b = init_deterministic(self.trace_size, self.context_size)
        self.a = self.param(
            "ffm_a",
            lambda key, shape: a,
            (),
        )
        self.b = self.param(
            "ffm_b",
            lambda key, shape: b,
            (),
        )

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

    def map_to_h(self, x: InputEmbedding) -> ScanInput:
        """Given an input embedding, this will map it to the format required for the associative scan."""
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        scan_input = jnp.repeat(jnp.expand_dims(gated_x, 3), self.context_size, axis=3)
        return scan_input

    def map_from_h(self, state: RecurrentState, x: InputEmbedding) -> chex.Array:
        """Given the recurrent state and the input embedding, this will map the recurrent state back to the output space."""
        T = state.shape[0]
        B = state.shape[1]
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(T, B, -1)
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        return out

    def log_gamma(self, t: Timestep) -> chex.Array:
        T = t.shape[0]
        B = t.shape[1]
        a = self.a
        b = self.b
        a = -jnp.abs(a).reshape((1, 1, self.trace_size, 1))
        b = b.reshape(1, 1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(T, B, 1, 1)

    def gamma(self, t: Timestep) -> chex.Array:
        return jnp.exp(self.log_gamma(t))

    def unwrapped_associative_update(
        self,
        carry: Tuple[RecurrentState, Timestep],
        incoming: Tuple[InputEmbedding, Timestep],
    ) -> Tuple[RecurrentState, Timestep]:
        (
            state,
            i,
        ) = carry
        x, j = incoming
        state = state * self.gamma(j) + x
        return state, j + i

    def wrapped_associative_update(
        self,
        carry: Tuple[Reset, RecurrentState, Timestep],
        incoming: Tuple[Reset, InputEmbedding, Timestep],
    ) -> Tuple[Reset, RecurrentState, Timestep]:
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
        x: InputEmbedding,
        state: RecurrentState,
        start: Reset,
    ) -> RecurrentState:
        """Given an input and recurrent state, this will update the recurrent state. This is equivalent
        to the inner-function g in the paper."""
        # x: [T, B, memory_size]
        # memory: [1, B, memory_size, context_size]
        T = x.shape[0]
        B = x.shape[1]
        timestep = jnp.ones((T + 1, B), dtype=jnp.int32).reshape(T + 1, B, 1, 1)
        # Add context dim
        start = start.reshape(T, B, 1, 1)

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

    @nn.compact
    def __call__(self, state: RecurrentState, inputs: Inputs) -> Tuple[RecurrentState, chex.Array]:

        # Add a sequence dimension to the recurrent state.
        state = jnp.expand_dims(state, 0)

        # Unpack inputs
        x, start = inputs

        # Map the input embedding to the recurrent state space.
        # This maps to the format required for the associative scan.
        scan_input = self.map_to_h(x)

        # Update the recurrent state
        state = self.scan(scan_input, state, start)

        # Map the recurrent state back to the output space
        out = self.map_from_h(state, x)

        # Take the final state of the sequence.
        final_state = state[-1:]

        # TODO: remove this when not running test
        out = nn.Dense(128)(out)
        out = nn.relu(out)
        out = nn.Dense(1)(out)

        # Remove the sequence dimemnsion from the final state.
        final_state = jnp.squeeze(final_state, 0)

        return final_state, out

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> RecurrentState:
        return jnp.zeros((batch_size, self.trace_size, self.context_size), dtype=jnp.complex64)


def train_memorize():

    USE_BATCH_VERSION = True  # Required to be true

    m = FFM(output_size=128, trace_size=64, context_size=4)

    batch_size = 16
    rem_ts = 10
    time_steps = rem_ts * 10
    obs_space = 8
    rng = jax.random.PRNGKey(0)
    if USE_BATCH_VERSION:
        x = jax.random.randint(rng, (time_steps, batch_size), 0, obs_space)
        y = jnp.stack(
            [
                jnp.repeat(x[::rem_ts, i], x.shape[0] // x[::rem_ts, i].shape[0])
                for i in range(batch_size)
            ],
            axis=-1,
        )
        x = x.reshape(time_steps, batch_size, 1)
        y = y.reshape(time_steps, batch_size, 1)

    start = jnp.zeros([time_steps, batch_size], dtype=bool).at[::rem_ts].set(True)

    s = m.initialize_carry(batch_size)

    params = m.init(jax.random.PRNGKey(0), s, (x, start))

    def error(params, x, start, key):
        s = m.initialize_carry(batch_size)

        # For BATCH VERSION
        if USE_BATCH_VERSION:
            x = jax.random.randint(rng, (time_steps, batch_size), 0, obs_space)
            y = jnp.stack(
                [
                    jnp.repeat(x[::rem_ts, i], x.shape[0] // x[::rem_ts, i].shape[0])
                    for i in range(batch_size)
                ],
                axis=-1,
            )
            x = x.reshape(time_steps, batch_size, 1)
            y = y.reshape(time_steps, batch_size, 1)

        final_state, y_hat = m.apply(params, s, (x, start))
        y_hat = jnp.squeeze(y_hat)
        y = jnp.squeeze(y)
        accuracy = (jnp.round(y_hat) == y).mean()
        loss = jnp.mean(jnp.abs(y - y_hat) ** 2)
        return loss, {"accuracy": accuracy, "loss": loss}

    optimizer = optax.adam(learning_rate=0.001)
    state = optimizer.init(params)
    loss_fn = jax.jit(jax.grad(error, has_aux=True))
    for step in range(10_000):
        rng = jax.random.split(rng)[0]
        grads, loss_info = loss_fn(params, x, start, rng)
        updates, state = jax.jit(optimizer.update)(grads, state)
        params = jax.jit(optax.apply_updates)(params, updates)
        print(f"Step {step+1}, Loss: {loss_info['loss']}, Accuracy: {loss_info['accuracy']}")


if __name__ == "__main__":
    # m = FFM(
    #     output_size=4,
    #     trace_size=5,
    #     context_size=6,
    # )
    # s = m.initialize_carry()
    # x = jnp.ones((10, 2))
    # start = jnp.zeros(10, dtype=bool)
    # params = m.init(jax.random.PRNGKey(0), x, s, start)
    # out = m.apply(params, x, s, start)

    # print(out)

    train_memorize()
