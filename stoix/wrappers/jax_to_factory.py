import threading
from typing import Optional

import jax
import numpy as np
from jumanji.env import Environment
from jumanji.specs import Spec
from jumanji.types import TimeStep

from stoix.utils.env_factory import EnvFactory


class JaxToStateful:
    """Converts a Stoix-ready JAX environment to a stateful one to be used by Sebulba systems."""

    def __init__(self, env: Environment, num_envs: int, device: jax.Device, init_seed: int):
        self.env = env
        self.num_envs = num_envs
        self.device = device

        # Create the metrics
        self.running_count_episode_return = np.zeros(self.num_envs, dtype=float)
        self.running_count_episode_length = np.zeros(self.num_envs, dtype=int)
        self.episode_return = np.zeros(self.num_envs, dtype=float)
        self.episode_length = np.zeros(self.num_envs, dtype=int)

        # Create the seeds
        max_int = np.iinfo(np.int32).max
        min_int = np.iinfo(np.int32).min
        init_seeds = jax.random.randint(
            jax.random.PRNGKey(init_seed), (num_envs,), min_int, max_int
        )
        self.rng_keys = jax.vmap(jax.random.PRNGKey)(init_seeds)

        # Vmap and compile the reset and step functions
        self.vmapped_reset = jax.jit(jax.vmap(self.env.reset), device=self.device)
        self.vmapped_step = jax.jit(jax.vmap(self.env.step, in_axes=(0, 0)), device=self.device)

    def reset(
        self, *, seed: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ) -> TimeStep:
        with jax.default_device(self.device):

            self.state, timestep = self.vmapped_reset(self.rng_keys)

            # Reset the metrics
            self.running_count_episode_return = np.zeros(self.num_envs, dtype=float)
            self.running_count_episode_length = np.zeros(self.num_envs, dtype=int)
            self.episode_return = np.zeros(self.num_envs, dtype=float)
            self.episode_length = np.zeros(self.num_envs, dtype=int)

            # Create the metrics dict
            metrics = {
                "episode_return": np.zeros(self.num_envs, dtype=float),
                "episode_length": np.zeros(self.num_envs, dtype=int),
                "is_terminal_step": np.zeros(self.num_envs, dtype=bool),
            }

            timestep_extras = timestep.extras

            timestep_extras["metrics"] = metrics

            timestep = timestep.replace(extras=timestep_extras)

        return timestep

    def step(self, action: list) -> TimeStep:
        with jax.default_device(self.device):
            self.state, timestep = self.vmapped_step(self.state, action)

            ep_done = timestep.last()
            not_done = ~ep_done

            # Counting episode return and length.
            new_episode_return = self.running_count_episode_return + timestep.reward
            new_episode_length = self.running_count_episode_length + 1

            # Update the episode return and length if the episode is done otherwise
            # keep the previous values
            episode_return_info = self.episode_return * not_done + new_episode_return * ep_done
            episode_length_info = self.episode_length * not_done + new_episode_length * ep_done
            # Update the running count
            self.running_count_episode_return = new_episode_return * not_done
            self.running_count_episode_length = new_episode_length * not_done

            self.episode_return = episode_return_info
            self.episode_length = episode_length_info

            # Create the metrics dict
            metrics = {
                "episode_return": episode_return_info,
                "episode_length": episode_length_info,
                "is_terminal_step": ep_done,
            }

            timestep_extras = timestep.extras
            timestep_extras["metrics"] = metrics
            timestep = timestep.replace(extras=timestep_extras)

        return timestep

    def observation_spec(self) -> Spec:
        return self.env.observation_spec()

    def action_spec(self) -> Spec:
        return self.env.action_spec()

    def close(self) -> None:
        pass


class JaxEnvFactory(EnvFactory):
    """
    Create environments using stoix-ready JAX environments
    """

    def __init__(self, jax_env: Environment, init_seed: int):
        self.jax_env = jax_env
        self.cpu = jax.devices("cpu")[0]
        self.seed = init_seed
        # a lock is needed because this object will be used from different threads.
        # We want to make sure all seeds are unique
        self.lock = threading.Lock()

    def __call__(self, num_envs: int) -> JaxToStateful:
        with self.lock:
            seed = self.seed
            self.seed += num_envs
            return JaxToStateful(self.jax_env, num_envs, self.cpu, seed)
