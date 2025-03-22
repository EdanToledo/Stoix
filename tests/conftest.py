import os
import sys
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple

# Ensure the stoix module is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def fixed_seed():
    """Fixture to provide a fixed seed for tests."""
    return 42

@pytest.fixture
def rng_key(fixed_seed):
    """Fixture to provide a JAX PRNG key."""
    return jax.random.PRNGKey(fixed_seed)

@pytest.fixture
def multiple_rng_keys(fixed_seed, n=5):
    """Fixture to provide multiple JAX PRNG keys."""
    keys = jax.random.split(jax.random.PRNGKey(fixed_seed), n)
    return keys 