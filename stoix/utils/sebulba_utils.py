import queue
import threading
import time
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from colorama import Fore, Style
from jax.sharding import Sharding
from jumanji.types import TimeStep

from stoix.base_types import Parameters, StoixTransition

QUEUE_PUT_TIMEOUT = 100


# Copied from https://github.com/instadeepai/sebulba/blob/main/sebulba/core.py
class ThreadLifetime:
    """Simple class for a mutable boolean that can be used to signal a thread to stop."""

    def __init__(self) -> None:
        self._stop = False

    def should_stop(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True


class OnPolicyPipeline(threading.Thread):
    """
    The `Pipeline` shards trajectories into `learner_devices`,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, learner_sharding: Sharding, lifetime: ThreadLifetime):
        """
        Initializes the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_sharding: The sharding used for the learner's update function.
            lifetime: A `ThreadLifetime` which is used to stop this thread.
        """
        super().__init__(name="Pipeline")

        self.sharding = learner_sharding
        self.tickets_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self.lifetime = lifetime

    def run(self) -> None:
        """This function ensures that trajectories on the queue are consumed in the right order. The
        start_condition and end_condition are used to ensure that only 1 thread is processing an
        item from the queue at one time, ensuring predictable memory usage.
        """
        while not self.lifetime.should_stop():
            try:
                start_condition, end_condition = self.tickets_queue.get(timeout=1)
                with end_condition:
                    with start_condition:
                        start_condition.notify()
                    end_condition.wait()
            except queue.Empty:
                continue

    def put(
        self,
        traj: Sequence[StoixTransition],
        timestep: TimeStep,
        actor_timings_dict: Dict[str, List[float]],
        actor_episode_metrics: List[Dict[str, List[float]]],
    ) -> None:
        """Put a trajectory on the queue to be consumed by the learner."""
        start_condition, end_condition = (threading.Condition(), threading.Condition())
        with start_condition:
            self.tickets_queue.put((start_condition, end_condition))
            start_condition.wait()  # wait to be allowed to start

        # [Transition(num_envs)] * rollout_len --> Transition[(num_envs, rollout_len,)
        traj = self.stack_trajectory(traj)
        traj, timestep = jax.device_put((traj, timestep), device=self.sharding)

        # Concatenate metrics - List[Dict[str, List[float]]] --> Dict[str, List[float]]
        actor_episode_metrics = self.concatenate_metrics(actor_episode_metrics)

        # We block on the put to ensure that actors wait for the learners to catch up. This does two
        # things:
        # 1. It ensures that the actors don't get too far ahead of the learners, which could lead to
        # off-policy data.
        # 2. It ensures that the actors don't in a sense "waste" samples and their time by
        # generating samples that the learners can't consume.
        # However, we put a timeout of 180 seconds to avoid deadlocks in case the learner
        # is not consuming the data. This is a safety measure and should not be hit in normal
        # operation. We use a try-finally since the lock has to be released even if an exception
        # is raised.
        try:
            self._queue.put(
                (traj, timestep, actor_timings_dict, actor_episode_metrics),
                block=True,
                timeout=QUEUE_PUT_TIMEOUT,
            )
        except queue.Full:
            print(
                f"{Fore.RED}{Style.BRIGHT}Pipeline is full and actor has timed out, "
                f"this should not happen. A deadlock might be occurring{Style.RESET_ALL}"
            )
        finally:
            with end_condition:
                end_condition.notify()  # notify that we have finished

    def qsize(self) -> int:
        """Returns the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get(
        self, block: bool = True, timeout: Union[float, None] = None
    ) -> Tuple[StoixTransition, TimeStep, Dict[str, List[float]], Dict[str, List[float]]]:
        """Get a trajectory from the pipeline."""
        return self._queue.get(block, timeout)  # type: ignore

    @partial(jax.jit, static_argnums=(0,))
    def stack_trajectory(self, trajectory: List[StoixTransition]) -> StoixTransition:
        """Stack a list of parallel_env transitions into a single
        transition of shape [num_envs, rollout_len, ...] i.e.
        stack to create the time axis."""
        return jax.tree_map(  # type: ignore
            lambda *x: jnp.stack(x, axis=1),
            *trajectory,
        )

    @partial(jax.jit, static_argnums=(0,))
    def concatenate_metrics(
        self, actor_metrics: List[Dict[str, List[float]]]
    ) -> Dict[str, List[float]]:
        """Concatenate a list of actor metrics into a single dictionary."""
        return jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *actor_metrics)  # type: ignore

    def clear(self) -> None:
        """Clear the pipeline."""
        while not self._queue.empty():
            try:
                self._queue.get(block=False)
            except queue.Empty:
                break


class ParamsSource(threading.Thread):
    """A `ParamSource` is a component that allows networks params to be passed from a
    `Learner` component to `Actor` components.
    """

    def __init__(self, init_value: Parameters, device: jax.Device, lifetime: ThreadLifetime):
        super().__init__(name=f"ParamsSource-{device.id}")
        self.value: Parameters = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue()
        self.lifetime = lifetime

    def run(self) -> None:
        """This function is responsible for updating the value of the `ParamSource` when a new value
        is available.
        """
        while not self.lifetime.should_stop():
            try:
                waiting = self.new_value.get(block=True, timeout=1)
                self.value = jax.device_put(waiting, self.device)
            except queue.Empty:
                continue

    def update(self, new_params: Parameters) -> None:
        """Update the value of the `ParamSource` with a new value.

        Args:
            new_params: The new value to update the `ParamSource` with.
        """
        self.new_value.put(new_params)

    def get(self) -> Parameters:
        """Get the current value of the `ParamSource`."""
        return self.value


class RecordTimeTo:
    def __init__(self, to: Any):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)
