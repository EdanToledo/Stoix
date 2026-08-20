"""SPO's particles must not share the environment's PRNG stream.

`make_root_fn` builds the search's starting point by copying the current environment
state once per particle. `broadcast_tree` copies it *verbatim*, PRNG key included,
and for a stochastic environment model that is wrong twice over:

  1. every particle then draws the *same* transition, so the search cannot average
     over the model's randomness at all. Algorithm 1 of the SPO paper samples
     `s_{i+1} ~ T_model(s_i, a_i)`; one shared key makes all N samples identical.
  2. that key belongs to the *live* environment, so the model reproduces exactly the
     transition that is about to happen. The search plans against a known future
     instead of an uncertain one.

Both tests go through `root_fn` itself, so removing the re-keying call from
`make_root_fn` makes them fail.

`2048` is used because it is stochastic (a new tile appears at random) and single
player, and because `pgx` is already a dependency of Stoix.
"""

from typing import Any, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import tensorflow_probability.substrates.jax as tfp
from absl.testing import absltest
from omegaconf import DictConfig, OmegaConf
from stoa.env_adapters.pgx import PGXToStoa

from stoix.systems.spo.ff_spo import make_root_fn

NUM_PARTICLES = 16
NUM_ACTIONS = 4


class _Online(NamedTuple):
    online: chex.ArrayTree


class _Params(NamedTuple):
    actor_params: _Online
    critic_params: _Online


def _config() -> DictConfig:
    return OmegaConf.create(
        {
            "system": {
                "num_particles": NUM_PARTICLES,
                "root_exploration_dirichlet_alpha": 0.3,
                "root_exploration_dirichlet_fraction": 0.0,
            }
        }
    )


def _actor_apply(_: chex.ArrayTree, observation: chex.ArrayTree) -> Any:
    """A uniform categorical policy: the search's prior is irrelevant here."""
    del observation
    return tfp.distributions.Categorical(logits=jnp.zeros((1, NUM_ACTIONS)))


def _critic_apply(_: chex.ArrayTree, observation: chex.ArrayTree) -> chex.Array:
    del observation
    return jnp.zeros((1,))


def _root_output() -> Tuple[Any, Any]:
    """Run `root_fn` on a batch-of-one 2048 state and return (env_state, particles)."""
    env = PGXToStoa(pgx.make("2048"))
    env_state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(0))
    env_state = jax.tree.map(lambda x: x[None], env_state)
    observation = jax.tree.map(lambda x: x[None], timestep.observation)

    root_fn = make_root_fn(_actor_apply, _critic_apply, _config())
    params = _Params(_Online(online={}), _Online(online={}))
    root = jax.jit(root_fn)(params, observation, env_state, jax.random.PRNGKey(1))
    return env, env_state, root.particle_env_states


def _distinct_transitions(env: Any, particle_states: chex.ArrayTree) -> int:
    """How many different boards the particles reach when they all play action 0."""
    flat = jax.tree.map(lambda x: x.reshape((NUM_PARTICLES,) + x.shape[2:]), particle_states)
    nxt, _ = jax.jit(jax.vmap(env.step))(flat, jnp.zeros((NUM_PARTICLES,), jnp.int32))
    boards = np.asarray(nxt.base_env_state._board)
    return len(np.unique(boards, axis=0))


class ParticleRNGTest(absltest.TestCase):
    def test_particles_get_independent_keys(self) -> None:
        """Each particle carries its own key, and none of them is the environment's."""
        _, env_state, particles = _root_output()
        keys = np.asarray(particles.rng_key).reshape(NUM_PARTICLES, -1)

        self.assertEqual(len(np.unique(keys, axis=0)), NUM_PARTICLES, "particles share a PRNG key")
        live = np.asarray(env_state.rng_key).reshape(-1)
        for i in range(NUM_PARTICLES):
            self.assertFalse(
                np.array_equal(keys[i], live),
                f"particle {i} kept the live environment's key",
            )

    def test_particles_see_different_transitions(self) -> None:
        """The model samples the transition instead of replaying a single draw.

        With the shared key every particle reaches the same board, so the count is 1.
        """
        env, _, particles = _root_output()
        self.assertGreater(
            _distinct_transitions(env, particles),
            1,
            "the search still sees a single sampled transition",
        )


if __name__ == "__main__":
    absltest.main()
