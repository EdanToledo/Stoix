"""Fitness shaping functions for Evolutionary Strategies.

All functions take a 1D array of fitnesses and return a 1D array of shaped fitnesses.
All operations use jnp for JIT compatibility.
"""

from typing import Callable

import chex
import jax.numpy as jnp


def centered_rank_shaping(fitnesses: chex.Array) -> chex.Array:
    """Rank-based fitness shaping that maps to uniform[-0.5, 0.5].

    Assigns ranks based on fitness values, then normalizes to [-0.5, 0.5].
    This is the standard fitness shaping used in OpenAI ES (Salimans et al., 2017).
    """
    n = fitnesses.shape[0]
    # argsort twice gives ranks
    sorted_indices = jnp.argsort(fitnesses)
    ranks = jnp.zeros_like(fitnesses)
    ranks = ranks.at[sorted_indices].set(jnp.arange(n, dtype=fitnesses.dtype))
    # Normalize to [-0.5, 0.5]
    return ranks / (n - 1) - 0.5


def z_score_shaping(fitnesses: chex.Array) -> chex.Array:
    """Z-score normalization: (f - mean) / std."""
    mean = jnp.mean(fitnesses)
    std = jnp.std(fitnesses) + 1e-8
    return (fitnesses - mean) / std


def raw_shaping(fitnesses: chex.Array) -> chex.Array:
    """Identity transform -- no fitness shaping applied."""
    return fitnesses


_FITNESS_SHAPING_FNS = {
    "centered_rank": centered_rank_shaping,
    "z_score": z_score_shaping,
    "raw": raw_shaping,
}


def get_fitness_shaping_fn(name: str) -> Callable[[chex.Array], chex.Array]:
    """Look up a fitness shaping function by name.

    Args:
        name: One of 'centered_rank', 'z_score', 'raw'.

    Returns:
        A JAX-compatible fitness shaping function.

    Raises:
        ValueError: If name is not recognized.
    """
    if name not in _FITNESS_SHAPING_FNS:
        raise ValueError(
            f"Unknown fitness shaping: '{name}'. "
            f"Available options: {list(_FITNESS_SHAPING_FNS.keys())}"
        )
    return _FITNESS_SHAPING_FNS[name]
