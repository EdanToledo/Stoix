from typing import Any, Dict, Optional

import numpy as np
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import StepType, TimeStep
from numpy.typing import NDArray

from stoix.base_types import Observation

NEXT_OBS_KEY_IN_EXTRAS = "next_obs"

class EnvPoolToJumanji:
    """Converts from the Gymnasium envpool API to Jumanji's API."""

    def __init__(self, env: Any):
        self.env = env
        obs, _ = self.env.reset()
        self.num_envs = obs.shape[0]
        self.obs_shape = obs.shape[1:]
        self.num_actions = self.env.action_space.n
        self._default_action_mask = np.ones((self.num_envs, self.num_actions), dtype=np.float32)

        # Create the metrics
        self.running_count_episode_return = np.zeros(self.num_envs, dtype=float)
        self.running_count_episode_length = np.zeros(self.num_envs, dtype=int)
        self.episode_return = np.zeros(self.num_envs, dtype=float)
        self.episode_length = np.zeros(self.num_envs, dtype=int)

        # See if the env has lives - Atari specific
        info = self.env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if "lives" in info and info["lives"].sum() > 0:
            print("Env has lives")
            self.has_lives = True
        else:
            self.has_lives = False
        self.env.close()

        # Set the flag to use the gym autoreset API
        self._use_gym_autoreset_api = True

    def reset(
        self, *, seed: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ) -> TimeStep:
        obs, info = self.env.reset()

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
        info[NEXT_OBS_KEY_IN_EXTRAS] = obs.copy()

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)
        ep_done = np.logical_or(terminated, truncated)
        not_done = 1 - ep_done
        info[NEXT_OBS_KEY_IN_EXTRAS] = obs.copy()
        if self._use_gym_autoreset_api:
            env_ids_to_reset = np.where(ep_done)[0]
            if len(env_ids_to_reset) > 0:
                (
                    reset_obs,
                    _,
                    _,
                    _,
                    _,
                ) = self.env.step(np.zeros_like(action), env_ids_to_reset)
                obs[env_ids_to_reset] = reset_obs
            

        # Counting episode return and length.
        if "reward" in info:
            metric_reward = info["reward"]
        else:
            metric_reward = rewards

        # Counting episode return and length.
        new_episode_return = self.running_count_episode_return + metric_reward
        new_episode_length = self.running_count_episode_length + 1

        # Previous episode return/length until done and then the next episode return.
        # If the env has lives (Atari), we only consider the return and length of the episode
        # every time all lives are exhausted.
        if self.has_lives:
            all_lives_exhausted = info["lives"] == 0
            not_all_lives_exhausted = 1 - all_lives_exhausted
            # Update the episode return and length if all lives are exhausted otherwise
            # keep the previous values
            episode_return_info = (
                self.episode_return * not_all_lives_exhausted
                + new_episode_return * all_lives_exhausted
            )
            episode_length_info = (
                self.episode_length * not_all_lives_exhausted
                + new_episode_length * all_lives_exhausted
            )
            # Update the running count
            self.running_count_episode_return = new_episode_return * not_all_lives_exhausted
            self.running_count_episode_length = new_episode_length * not_all_lives_exhausted
        else:
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
        info["metrics"] = metrics

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
        extras[NEXT_OBS_KEY_IN_EXTRAS] = self._format_observation(info[NEXT_OBS_KEY_IN_EXTRAS], info)
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
        return DiscreteArray(num_values=self.num_actions)

    def close(self) -> None:
        self.env.close()
