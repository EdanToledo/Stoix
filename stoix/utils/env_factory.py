import abc
import threading
from typing import Any

import envpool


class EnvFactory(abc.ABC):
    """
    Abstract class to create environments
    """

    @abc.abstractmethod
    def __call__(self, num_envs: int) -> Any:
        pass


class EnvPoolFactory(EnvFactory):
    """
    Create environments with different seeds for each `Actor`
    """

    def __init__(self, init_seed: int = 42, **kwargs: Any):
        self.seed = init_seed
        # a lock is needed because this object will be used from different threads.
        # We want to make sure all seeds are unique
        self.lock = threading.Lock()
        self.kwargs = kwargs

    def __call__(self, num_envs: int) -> Any:
        with self.lock:
            seed = self.seed
            self.seed += num_envs
            return envpool.make(**self.kwargs, num_envs=num_envs, seed=seed)