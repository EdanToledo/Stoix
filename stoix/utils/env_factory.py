import abc
import threading
from typing import Any

import envpool
import gymnasium
from omegaconf import DictConfig

from stoix.wrappers.envpool import EnvPoolToJumanji
from stoix.wrappers.gymnasium import GymRecordEpisodeMetrics, GymToJumanji, GymnasiumWrapper


class EnvFactory(abc.ABC):
    """
    Abstract class to create environments
    """
    
    def __init__(self, task_id : str, init_seed: int = 42, **kwargs: Any):
        self.task_id = task_id
        self.seed = init_seed
        # a lock is needed because this object will be used from different threads.
        # We want to make sure all seeds are unique
        self.lock = threading.Lock()
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, num_envs: int) -> Any:
        pass


class EnvPoolFactory(EnvFactory):
    """
    Create environments with different seeds for each `Actor`
    """

    def __call__(self, num_envs: int) -> Any:
        with self.lock:
            seed = self.seed
            self.seed += num_envs
            return EnvPoolToJumanji(envpool.make(task_id=self.task_id, env_type="gymnasium", num_envs=num_envs, seed=seed, gym_reset_return_info=True, **self.kwargs))

class GymnasiumFactory(EnvFactory):
    """
    Create environments using gymnasium
    """
    
    def create_single_gym_env(self) -> gymnasium.Env:
        env = gymnasium.make(id=self.task_id, **self.kwargs)
        env = GymnasiumWrapper(env)
        env = GymRecordEpisodeMetrics(env)
        return env
    
    def __call__(self, num_envs: int) -> Any:
        with self.lock:
            envs = gymnasium.vector.AsyncVectorEnv(
                [self.create_single_gym_env for _ in range(num_envs)],
            )
            return GymToJumanji(envs)
