import jax
import pytest


@pytest.fixture
def fixed_seed() -> int:
    """Fixture to provide a fixed seed for tests."""
    return 42


@pytest.fixture
def rng_key(fixed_seed: int) -> jax.Array:
    """Fixture to provide a JAX PRNG key."""
    return jax.random.PRNGKey(fixed_seed)


@pytest.fixture
def multiple_rng_keys(fixed_seed: int, n: int = 5) -> jax.Array:
    """Fixture to provide multiple JAX PRNG keys."""
    keys = jax.random.split(jax.random.PRNGKey(fixed_seed), n)
    return keys
