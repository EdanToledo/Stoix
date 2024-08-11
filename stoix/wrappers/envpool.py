import sys
import traceback
import warnings
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Optional, Tuple, Union

import envpool
import jax
import numpy as np
from jumanji.types import StepType, TimeStep
from numpy.typing import NDArray

from stoix.base_types import Observation


class EnvPoolToJumanji:
    """Converts from the Gym API to the dm_env API, using Jumanji's Timestep type."""

    def __init__(self, env: Any):
        self.env = env
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self._default_action_mask = np.ones(self.num_actions, dtype=np.float32)
    
    def reset(
        self, seed: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ) -> TimeStep:
        obs, info = self.env.reset()

        ep_done = np.zeros(self.num_envs, dtype=float)
        rewards = np.zeros(self.num_envs, dtype=float)
        terminated = np.zeros(self.num_envs, dtype=float)

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)

        ep_done = np.logical_or(terminated, truncated)

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def _format_observation(self, obs: NDArray, info: Dict) -> Observation:
        action_mask = self._default_action_mask
        multi_env_action_mask = np.stack([action_mask] * obs.shape[0])
        return Observation(agent_view=obs, action_mask=multi_env_action_mask)

    def _create_timestep(
        self, obs: NDArray, ep_done: NDArray, terminated: NDArray, rewards: NDArray, info: Dict
    ) -> TimeStep:
        obs = self._format_observation(obs, info)
        extras = {} #jax.tree.map(lambda *x: np.stack(x), *info["metrics"])
        step_type = np.where(ep_done, StepType.LAST, StepType.MID)

        return TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - terminated,
            observation=obs,
            extras=extras,
        )
