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


class GymnasiumWrapper(gymnasium.Wrapper):
    """Base wrapper for gymnasium environments."""

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
        if isinstance(self._env.action_space, gymnasium.spaces.Discrete):
            self.num_actions = self._env.action_space.n
        else:
            self.num_actions = self._env.action_space.shape[0]

        self.default_action_mask = np.ones(self.num_actions, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[NDArray, Dict]:

        agents_view, info = self._env.reset(seed=seed, options=options)

        return np.array(agents_view), info

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        agents_view, reward, terminated, truncated, info = self._env.step(actions)

        agents_view = np.asarray(agents_view)
        reward = np.asarray(reward)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)

        return agents_view, reward, terminated, truncated, info


class GymRecordEpisodeMetrics(gymnasium.Wrapper):
    """Record the episode returns and lengths."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._env = env
        self.running_count_episode_return = 0.0
        self.running_count_episode_length = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[NDArray, Dict]:
        agents_view, info = self._env.reset(seed=seed, options=options)

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

        self.running_count_episode_return += reward
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
    
    def __init__(self, env: gymnasium.vector.AsyncVectorEnv):
        super().__init__(env)
        self.env = env
        self.num_envs = self.env.num_envs
        if isinstance(self.env.single_action_space, gymnasium.spaces.Discrete):
            self.num_actions = self.env.single_action_space.n
        else:
            self.num_actions = self.env.single_action_space.shape[0]
        self.obs_shape = self.env.single_observation_space.shape
        self.default_action_mask = np.ones((self.num_envs, self.num_actions), dtype=np.float32)
    
    def reset(
        self, seed: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ) -> TimeStep:
        obs, info = self.env.reset(seed=seed, options=options)

        num_envs = int(self.env.num_envs)

        ep_done = np.zeros(num_envs, dtype=float)
        rewards = np.zeros(num_envs, dtype=float)
        terminated = np.zeros(num_envs, dtype=float)

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)

        ep_done = np.logical_or(terminated, truncated)

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def _format_observation(self, obs: NDArray, info: Dict) -> Observation:
        action_mask = self.default_action_mask
        return Observation(agent_view=obs, action_mask=action_mask)

    def _create_timestep(
        self, obs: NDArray, ep_done: NDArray, terminated: NDArray, rewards: NDArray, info: Dict
    ) -> TimeStep:
        obs = self._format_observation(obs, info)
        extras = jax.tree.map(lambda *x: np.stack(x), *info["metrics"])
        step_type = np.where(ep_done, StepType.LAST, StepType.MID)

        return TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - terminated,
            observation=obs,
            extras=extras,
        )
