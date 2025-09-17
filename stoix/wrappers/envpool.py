from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from stoa import ArraySpace, DiscreteSpace, Space, StepType, TimeStep


class EnvPoolToStoa:
    """Converts from envpool API to Stoa's API."""

    def __init__(self, env: Any):
        self.env = env
        obs, _ = self.env.reset()
        self.num_envs = obs.shape[0]
        self.obs_shape = obs.shape[1:]
        self.num_actions = self.env.action_space.n

        # Create the metrics
        self.running_count_episode_return = np.zeros(self.num_envs, dtype=float)
        self.running_count_episode_length = np.zeros(self.num_envs, dtype=int)
        self.episode_return = np.zeros(self.num_envs, dtype=float)
        self.episode_length = np.zeros(self.num_envs, dtype=int)

        # See if the env has lives - Atari specific
        info = self.env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if "lives" in info and info["lives"].sum() > 0:
            self.has_lives = True
            print(
                "This environment has lives. The episode return and "
                "length will be counted only when all lives are exhausted."
            )
        else:
            self.has_lives = False
        self.env.close()

        # Set the flag to use the stoix autoreset API
        # since envpool does auto resetting slightly differently
        self._use_autoreset_api = True

        self.max_episode_steps = self.env.spec.config.max_episode_steps

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

        timestep = self._create_timestep(obs, ep_done, terminated, rewards, info)

        return timestep

    def step(self, action: list) -> TimeStep:
        obs, rewards, terminated, truncated, info = self.env.step(action)
        truncated = info["elapsed_step"] >= self.max_episode_steps
        ep_done = np.logical_or(terminated, truncated)
        not_done = 1 - ep_done
        if self._use_autoreset_api:
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

    def _create_timestep(
        self, obs: NDArray, ep_done: NDArray, terminated: NDArray, rewards: NDArray, info: Dict
    ) -> TimeStep:
        step_type = np.where(ep_done, StepType.TERMINATED, StepType.MID)
        truncated = info["elapsed_step"] >= self.max_episode_steps
        discount = 1.0 - terminated
        discount = np.where(truncated, 1.0, discount)
        step_type = np.where(truncated, StepType.TRUNCATED, step_type)

        return TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=discount,
            observation=obs,
            extras=info,
        )

    def observation_space(self) -> Space:
        return ArraySpace(shape=self.obs_shape, dtype=float)

    def action_space(self) -> Space:
        return DiscreteSpace(num_values=self.num_actions)

    def close(self) -> None:
        self.env.close()
