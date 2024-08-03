import sys
import traceback
import warnings
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium
import jax
import numpy as np
from jumanji.types import StepType, TimeStep
from numpy.typing import NDArray

from stoix.base_types import Observation

# Filter out the warnings
warnings.filterwarnings("ignore", module="gymnasium.utils.passive_env_checker")


class GymWrapper(gymnasium.Wrapper):
    """Base wrapper for gym environments."""

    def __init__(
        self,
        env: gymnasium.Env,
    ):
        """Initialise the gym wrapper
        Args:
            env (gymnasium.env): gymnasium env instance.
        """
        super().__init__(env)
        self._env = env
        self.num_actions = self._env.action_space[0].n

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:
        if seed is not None:
            self.env.seed(seed)

        agents_view, info = self._env.reset()

        info = {"actions_mask": self.get_actions_mask(info)}

        return np.array(agents_view), info

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        info = {"actions_mask": self.get_actions_mask(info)}
        
        reward = np.array(reward)

        return agents_view, reward, terminated, truncated, info

    def get_actions_mask(self, info: Dict) -> NDArray:
        if "action_mask" in info:
            return np.array(info["action_mask"])
        return np.ones((self.num_agents, self.num_actions), dtype=np.float32)

class GymRecordEpisodeMetrics(gymnasium.Wrapper):
    """Record the episode returns and lengths."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._env = env
        self.running_count_episode_return = 0.0
        self.running_count_episode_length = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[NDArray, Dict]:
        agents_view, info = self._env.reset(seed, options)

        # Create the metrics dict
        metrics = {
            "episode_return": self.running_count_episode_return,
            "episode_length": self.running_count_episode_length,
            "is_terminal_step": True,
        }

        # Reset the metrics
        self.running_count_episode_return = 0.0
        self.running_count_episode_length = 0

        if "won_episode" in info:
            metrics["won_episode"] = info["won_episode"]

        info["metrics"] = metrics

        return agents_view, info

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        self.running_count_episode_return += float(np.mean(reward))
        self.running_count_episode_length += 1

        metrics = {
            "episode_return": self.running_count_episode_return,
            "episode_length": self.running_count_episode_length,
            "is_terminal_step": False,
        }
        if "won_episode" in info:
            metrics["won_episode"] = info["won_episode"]

        info["metrics"] = metrics

        return agents_view, reward, terminated, truncated, info


class GymToJumanji(gymnasium.Wrapper):
    """Converts from the Gym API to the dm_env API, using Jumanji's Timestep type."""

    def reset(
        self, seed: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ) -> TimeStep:
        obs, info = self.env.reset(seed=seed, options=options)
        
        num_envs = self.env.num_envs

        ep_done = np.zeros(num_envs, dtype=float)
        rewards = np.zeros((num_envs,), dtype=float)
        teminated = np.zeros((num_envs,), dtype=float)
        self.step_count = np.zeros((num_envs,), dtype=float)

        timestep = self._create_timestep(obs, ep_done, teminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)

        ep_done = np.logical_or(terminated, truncated).all(axis=1)
        self.step_count += 1

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def _format_observation(
        self, obs: NDArray, info: Dict
    ) -> Observation:
        """Create an observation from the raw observation and environment state."""

        obs = np.array(obs)
        action_mask = info["actions_mask"]
        obs_data = {"agents_view": obs, "action_mask": action_mask, "step_count": self.step_count}

        return Observation(**obs_data)

    def _create_timestep(
        self, obs: NDArray, ep_done: NDArray, terminated: NDArray, rewards: NDArray, info: Dict
    ) -> TimeStep:
        obs = self._format_observation(obs, info)
        extras = jax.tree.map(lambda *x: np.stack(x), *info["metrics"])
        step_type = np.where(ep_done, StepType.LAST, StepType.MID)
        terminated = np.all(terminated, axis=1)

        return TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - terminated,
            observation=obs,
            extras=extras,
        )