from functools import partial
from typing import Any, Dict, Tuple

import chex
import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax import vmap


def init_deterministic(
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 1_000
) -> Tuple[jax.Array, jax.Array]:
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
        self.a = self.param(
            "ffm_a",
            lambda key, shape: init_deterministic(self.trace_size, self.context_size)[0],
            (),
        )
        self.b = self.param(
            "ffm_b",
            lambda key, shape: init_deterministic(self.trace_size, self.context_size)[1],
            (),
        )

    @nn.compact
    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:

        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.trace_size * 2)(x)

        gate_in = Gate(self.trace_size)(x)
        pre = Gate(self.trace_size)(x)
        gated_x = pre * gate_in
        scan_input = jnp.repeat(jnp.expand_dims(gated_x, 2), self.context_size, axis=2)
        state = self.scan(scan_input, state, start)
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(
            state.shape[0], -1
        )
        z = nn.Dense(64)(z_in)
        gate_out = Gate(64)(x)
        skip = nn.Dense(64)(x)
        out = nn.LayerNorm(use_scale=False, use_bias=False)(z * gate_out) + skip * (1 - gate_out)
        final_state = state[-1:]

        out = nn.Dense(64)(out)
        out = nn.relu(out)
        out = nn.Dense(self.output_size)(out)

        return out, final_state

    def initial_state(self) -> jax.Array:
        return jnp.zeros((1, self.trace_size, self.context_size), dtype=jnp.complex64)

    def log_gamma(self, t: jax.Array) -> jax.Array:
        a = self.a
        b = self.b
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
        (
            state,
            i,
        ) = carry
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
        # timestep = jnp.arange(T + 1, dtype=jnp.int32)
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


def train_memorize():

    USE_BATCH_VERSION = True

    if USE_BATCH_VERSION:

        m = nn.vmap(
            FFM, in_axes=1, out_axes=1, variable_axes={"params": None}, split_rngs={"params": None}
        )(output_size=1, trace_size=64, context_size=4)
    else:
        m = FFM(output_size=1, trace_size=64, context_size=4)

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

    else:
        x = jax.random.randint(rng, (time_steps, batch_size), 0, obs_space).reshape(-1, 1)
        y = jnp.repeat(x[::rem_ts], x.shape[0] // x[::rem_ts].shape[0]).reshape(-1, 1)

    start = jnp.zeros([time_steps, batch_size], dtype=bool).at[::rem_ts].set(True)

    s = m.initial_state()

    # FOR BATCH VERSION
    if USE_BATCH_VERSION:
        s = jnp.expand_dims(s, 1)
        s = jnp.repeat(s, batch_size, axis=1)
    params = m.init(jax.random.PRNGKey(0), x, s, start)

    def error(params, x, start, key):
        s = m.initial_state()

        if USE_BATCH_VERSION:
            s = jnp.expand_dims(s, 1)
            s = jnp.repeat(s, batch_size, axis=1)

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
        else:
            x = jax.random.randint(key, (time_steps, batch_size), 0, obs_space).reshape(-1, 1)
            y = jnp.repeat(x[::rem_ts], x.shape[0] // x[::rem_ts].shape[0]).reshape(-1, 1)

        y_hat, final_state = m.apply(params, x, s, start)
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
    # s = m.initial_state()
    # x = jnp.ones((10, 2))
    # start = jnp.zeros(10, dtype=bool)
    # params = m.init(jax.random.PRNGKey(0), x, s, start)
    # out = m.apply(params, x, s, start)

    # print(out)

    train_memorize()
