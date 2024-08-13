from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium
import jax
import numpy as np
from jumanji.types import StepType, TimeStep
from numpy.typing import NDArray
from jumanji.specs import Array, Spec, DiscreteArray

from stoix.base_types import Observation


class VecGymToJumanji:
    """Converts from a Vectorised Gymnasium environment to Jumanji's API."""
    
    def __init__(self, env: gymnasium.vector.AsyncVectorEnv):
        self.env = env
        self.num_envs = int(self.env.num_envs)
        if isinstance(self.env.single_action_space, gymnasium.spaces.Discrete):
            self.num_actions = self.env.single_action_space.n
            self.discrete = True
        else:
            self.num_actions = self.env.single_action_space.shape[0]
            self.discrete = False
        self.obs_shape = self.env.single_observation_space.shape
        self._default_action_mask = np.ones((self.num_envs, self.num_actions), dtype=np.float32)

        # Create the metrics
        self.running_count_episode_return = np.zeros(self.num_envs, dtype=float)
        self.running_count_episode_length = np.zeros(self.num_envs, dtype=int)
        self.episode_return = np.zeros(self.num_envs, dtype=float)
        self.episode_length = np.zeros(self.num_envs, dtype=int)
        
    def reset(
        self, *, seed: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ) -> TimeStep:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs)
        ep_done = np.zeros(self.num_envs, dtype=float)
        rewards = np.zeros(self.num_envs, dtype=float)
        terminated = np.zeros(self.num_envs, dtype=float)
        
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
        
        info["metrics"] = metrics

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs)
        rewards = np.asarray(rewards)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        ep_done = np.logical_or(terminated, truncated)
        not_done = 1 - ep_done
        
        # Counting episode return and length.
        new_episode_return = self.running_count_episode_return + rewards
        new_episode_length = self.running_count_episode_length + 1

        # Previous episode return/length until done and then the next episode return.
        episode_return_info = self.episode_return * not_done + new_episode_return * ep_done
        episode_length_info = self.episode_length * not_done + new_episode_length * ep_done

        metrics = {
            "episode_return": episode_return_info,
            "episode_length": episode_length_info,
            "is_terminal_step": ep_done,
        }
        info["metrics"] = metrics

        # Update the metrics
        self.running_count_episode_return=new_episode_return * not_done
        self.running_count_episode_length=new_episode_length * not_done
        self.episode_return=episode_return_info
        self.episode_length=episode_length_info

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def _format_observation(self, obs: NDArray, info: Dict) -> Observation:
        action_mask = self._default_action_mask
        return Observation(agent_view=obs, action_mask=action_mask)

    def _create_timestep(
        self, obs: NDArray, ep_done: NDArray, terminated: NDArray, rewards: NDArray, info: Dict
    ) -> TimeStep:
        obs = self._format_observation(obs, info)
        extras = info["metrics"]
        step_type = np.where(ep_done, StepType.LAST, StepType.MID)

        return TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - terminated,
            observation=obs,
            extras=extras,
        )
        
    def observation_spec(self) -> Spec:
        agent_view_spec = Array(shape=self.obs_shape, dtype=float)
        return Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=Array(shape=(self.num_actions,), dtype=float),
            step_count=Array(shape=(), dtype=int),
        )

    def action_spec(self) -> Spec:
        if self.discrete:
            return DiscreteArray(num_values=self.num_actions)
        else:
            return Array(shape=(self.num_actions,), dtype=float)

    def close(self):
        self.env.close()