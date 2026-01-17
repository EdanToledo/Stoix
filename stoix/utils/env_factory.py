"""Environment factory utilities for Stoix.

This is a minimal fork that removes EnvPool support (not compatible with JAX 0.7+).
Only the abstract EnvFactory and GymnasiumFactory are kept.
"""

import abc
import threading
from typing import Any, Callable

import gymnasium

from stoix.wrappers.gymnasium import VecGymToStoa


class EnvFactory(abc.ABC):
    """
    Abstract class to create environments
    """

    def __init__(
        self,
        task_id: str,
        init_seed: int = 42,
        apply_wrapper_fn: Callable = lambda x: x,
        **kwargs: Any,
    ):
        self.task_id = task_id
        self.seed = init_seed
        self.apply_wrapper_fn = apply_wrapper_fn
        # a lock is needed because this object will be used from different threads.
        # We want to make sure all seeds are unique
        self.lock = threading.Lock()
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, num_envs: int) -> Any:
        pass


class GymnasiumFactory(EnvFactory):
    """
    Create environments using gymnasium
    """

    def __call__(self, num_envs: int) -> Any:
        with self.lock:
            vec_env = gymnasium.make_vec(
                id=self.task_id,
                num_envs=num_envs,
                vectorization_mode="sync",
                vector_kwargs={"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP},
                **self.kwargs,
            )
            return self.apply_wrapper_fn(VecGymToStoa(vec_env))
