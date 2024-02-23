from typing import Tuple

import chex
import jax
import jax.numpy as jnp


def calculate_gae(
    v_t: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    bootstrap_val: chex.Array,
    gae_lambda: float,
) -> Tuple[chex.Array, chex.Array]:
    """Calculate the Generalized Advantage Estimation (GAE) for a batch of trajectories.
    Trajectories are assumed to have the sequence dimension as the first dimension."""

    def _get_advantages(gae_and_next_value: Tuple, transition: Tuple[chex.Array, chex.Array, chex.Array]) -> Tuple:
        """Calculate the GAE for a single transition."""
        gae, next_value = gae_and_next_value

        r_t, d_t, v_t = transition

        delta = r_t + d_t * next_value - v_t
        gae = delta + gae_lambda * d_t * gae
        return (gae, v_t), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(bootstrap_val), bootstrap_val),
        (r_t, d_t, v_t),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + v_t
