import abc
import threading
from typing import Any, Callable

from colorama import Fore, Style

# Envpool is not usable on certain platforms, so we need to handle the ImportError
try:
    import envpool
except ImportError:
    envpool = None
    print(
        f"{Fore.MAGENTA}{Style.BRIGHT}Envpool not installed. "
        f"Please install it if you want to use the Envpool factory{Style.RESET_ALL}"
    )

import gymnasium

from stoix.wrappers.envpool import EnvPoolToStoa
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


class EnvPoolFactory(EnvFactory):
    """
    Create environments with different seeds for each `Actor`
    """

    def __call__(self, num_envs: int) -> Any:
        with self.lock:
            seed = self.seed
            self.seed += num_envs
            return self.apply_wrapper_fn(
                EnvPoolToStoa(
                    envpool.make(
                        task_id=self.task_id,
                        env_type="gymnasium",
                        num_envs=num_envs,
                        seed=seed,
                        gym_reset_return_info=True,
                        **self.kwargs,
                    )
                )
            )


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
