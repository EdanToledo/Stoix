import copy
import queue
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from omegaconf import DictConfig

from stoix.base_types import Parameters
from stoix.utils.logger import StoixLogger


class ThreadLifetime:
    """Manages thread lifecycle with stop signaling."""

    def __init__(self, thread_name: str, thread_id: int) -> None:
        self._stop = False
        self.thread_name = thread_name
        self.thread_id = thread_id

    @property
    def name(self) -> str:
        return self.thread_name

    @property
    def id(self) -> int:
        return self.thread_id

    def __repr__(self) -> str:
        return (
            f"Thread(thread_name={self.thread_name}, thread_id={self.thread_id}, stop={self._stop})"
        )

    def should_stop(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True


class OnPolicyPipeline:
    """Handles rollout communication for on-policy distributed RL."""

    def __init__(self, total_num_actors: int, queue_maxsize: int = 1):
        self.num_actors = total_num_actors

        # Create dedicated queues for each actor's rollout data
        self.rollout_queues: List[queue.Queue] = []
        for _ in range(total_num_actors):
            self.rollout_queues.append(queue.Queue(maxsize=queue_maxsize))

    def send_rollout(
        self, actor_idx: int, rollout_data: Tuple[int, int, Any], timeout: Optional[float] = None
    ) -> bool:
        """Send rollout data from actor."""
        try:
            if timeout is not None:
                self.rollout_queues[actor_idx].put(rollout_data, timeout=timeout)
            else:
                self.rollout_queues[actor_idx].put(rollout_data)
            return True
        except queue.Full:
            return False

    def collect_rollouts(self, timeout: Optional[float] = None) -> List[Tuple[int, int, Any]]:
        """Collect rollout data from all actors."""
        collected_data = []

        # Must collect from all actors to maintain synchronization
        for actor_idx in range(self.num_actors):
            try:
                if timeout is not None:
                    data = self.rollout_queues[actor_idx].get(timeout=timeout)
                else:
                    data = self.rollout_queues[actor_idx].get()
                collected_data.append(data)
            except queue.Empty:
                raise RuntimeError(f"Failed to collect rollout from actor {actor_idx}")

        return collected_data

    def clear_all_queues(self) -> None:
        """Clear all rollout queues."""
        for rollout_queue in self.rollout_queues:
            while not rollout_queue.empty():
                try:
                    rollout_queue.get_nowait()
                except queue.Empty:
                    break


class ParameterServer:
    """Handles parameter distribution for distributed RL."""

    def __init__(
        self,
        total_num_actors: int,
        actor_devices: Sequence[jax.Device],
        actors_per_device: int,
        queue_maxsize: int = 1,
    ):
        self.num_actors = total_num_actors
        self.actor_devices = actor_devices
        self.actors_per_device = actors_per_device

        # Create dedicated queues for each actor's parameters
        self.param_queues: List[queue.Queue] = []
        for _ in range(total_num_actors):
            self.param_queues.append(queue.Queue(maxsize=queue_maxsize))

    def distribute_params(
        self,
        params: Parameters,
        block: bool = True,
        timeout: Optional[float] = None,
        block_params_until_ready: bool = False,
    ) -> None:
        """Distribute parameters to all actors with device placement."""
        # Convert from replicated to single copy for distribution
        try:
            unreplicated_params = flax.jax_utils.unreplicate(params)
        except Exception as e:
            warnings.warn(f"Failed to unreplicate parameters: {e}", stacklevel=2)
            return

        # Place parameters on each actor device and distribute to threads
        actor_idx = 0
        for _device_idx, device in enumerate(self.actor_devices):
            try:
                # Ensure parameters are properly placed on target device
                device_params = jax.device_put(unreplicated_params, device)
                if block_params_until_ready:
                    device_params = jax.block_until_ready(device_params)

                # Distribute to all actors on this device
                for _ in range(self.actors_per_device):
                    try:
                        if block:
                            if timeout is not None:
                                self.param_queues[actor_idx].put(device_params, timeout=timeout)
                            else:
                                self.param_queues[actor_idx].put(device_params)
                        else:
                            self.param_queues[actor_idx].put_nowait(device_params)
                    except (queue.Full, queue.Empty):
                        warnings.warn(
                            f"Failed to put parameters in queue {actor_idx}", stacklevel=2
                        )
                    actor_idx += 1
            except Exception as e:
                warnings.warn(f"Failed to place parameters on device {device}: {e}", stacklevel=2)
                # Skip actors on this device
                actor_idx += self.actors_per_device

    def get_params(self, actor_idx: int, timeout: Optional[float] = None) -> Optional[Parameters]:
        """Get parameters for an actor."""
        try:
            if timeout is not None:
                params = self.param_queues[actor_idx].get(timeout=timeout)
            else:
                params = self.param_queues[actor_idx].get()

            if params is None:  # Shutdown signal
                return None

            # Ensure parameters are ready for immediate use
            return jax.block_until_ready(params)
        except queue.Empty:
            return None

    def shutdown_actors(self) -> None:
        """Send shutdown signals to all actors."""
        for param_queue in self.param_queues:
            try:
                param_queue.put_nowait(None)  # Shutdown sentinel
            except queue.Full:
                pass  # Actor will eventually check lifetime

    def clear_all_queues(self) -> None:
        """Clear all parameter queues."""
        for param_queue in self.param_queues:
            while not param_queue.empty():
                try:
                    param_queue.get_nowait()
                except queue.Empty:
                    break


class AsyncEvaluatorBase(threading.Thread, ABC):
    """Base class for asynchronous evaluators."""

    def __init__(
        self,
        evaluator: Callable,
        logger: StoixLogger,
        config: DictConfig,
        checkpointer: Any,
        save_checkpoint: bool,
        lifetime: ThreadLifetime,
    ):
        super().__init__(name="AsyncEvaluator")
        self.evaluator = evaluator
        self.logger = logger
        self.config = config
        self.checkpointer = checkpointer
        self.save_checkpoint = save_checkpoint
        self.lifetime = lifetime

        # Evaluation queue and tracking
        self.eval_queue: queue.Queue = queue.Queue()
        self.max_episode_return = jnp.float32(-1e7)
        self.best_params = None
        self.eval_step = 0

        # Progress tracking for completion signaling
        self.expected_evaluations = config.arch.num_evaluation
        self.completed_evaluations = 0
        self._evaluation_lock = threading.Lock()
        self._all_evaluations_done = threading.Event()
        self._eval_metrics: List[Dict[str, Any]] = []

    @abstractmethod
    def run(self) -> None:
        """Run the evaluation loop. Must be implemented by subclasses."""
        pass

    def submit_evaluation(
        self,
        learner_state: Any,
        eval_key: chex.PRNGKey,
        eval_step: int,
        global_step_count: int,
    ) -> None:
        """Submit evaluation data for async processing."""
        try:
            self.eval_queue.put_nowait((learner_state, eval_key, eval_step, global_step_count))
            print(
                f"{Fore.YELLOW}{Style.BRIGHT}Submitted evaluation "
                f"{eval_step+1}/{self.expected_evaluations}, Current Eval Queue "
                f"Size: {self.eval_queue.qsize()}{Style.RESET_ALL}"
            )
        except queue.Full:
            # Skip evaluation to avoid blocking the learner
            warnings.warn("Evaluation queue is full, skipping evaluation", stacklevel=2)

    def get_best_params(self) -> Any:
        """Get the best parameters found so far."""
        return self.best_params

    def wait_for_all_evaluations(self, timeout: float = 300.0) -> bool:
        """Wait for all evaluations to complete."""
        if self.expected_evaluations <= 0:
            return True  # No evaluations expected
        return self._all_evaluations_done.wait(timeout)

    def shutdown(self) -> None:
        """Shutdown the evaluator."""
        try:
            self.eval_queue.put_nowait(None)  # Shutdown sentinel
        except queue.Full:
            pass

    def _update_evaluation_progress(self) -> None:
        """Update evaluation progress and signal completion."""
        self.eval_step += 1

        # Thread-safe progress tracking
        with self._evaluation_lock:
            self.completed_evaluations += 1
            if self.completed_evaluations >= self.expected_evaluations:
                self._all_evaluations_done.set()

    def _update_best_params(self, episode_return: Any, actor_params: Any) -> None:
        """Update best parameters if current episode return is better."""
        if self.config.arch.absolute_metric and self.max_episode_return <= episode_return:
            self.best_params = copy.deepcopy(actor_params)
            self.max_episode_return = episode_return

    def get_eval_metrics(self) -> List[Dict[str, Any]]:
        """Get evaluation metrics collected so far."""
        return self._eval_metrics

    def add_eval_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add evaluation metrics to the list."""
        with self._evaluation_lock:
            self._eval_metrics.append(metrics)
            if len(self._eval_metrics) > self.expected_evaluations:
                self._eval_metrics.pop(0)

    def get_final_episode_return(self) -> float:
        """Get the final episode return from the last evaluation."""
        if self._eval_metrics:
            return float(np.mean(self._eval_metrics[-1].get("episode_return", 0.0)).item())
        return float(0.0)


def tree_stack_numpy(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Stack arrays in list of dicts into single dict with concatenated arrays."""
    if not list_of_dicts:
        return {}

    result = {}
    keys = list_of_dicts[0].keys()

    # Process each key across all dictionaries
    for key in keys:
        arrays_to_concat = []

        for d in list_of_dicts:
            value = d[key]
            # Convert various types to numpy arrays for concatenation
            if isinstance(value, np.ndarray):
                arrays_to_concat.append(value)
            elif isinstance(value, (list, tuple)):
                arrays_to_concat.append(np.array(value))
            else:
                arrays_to_concat.append(np.array([value]))

        result[key] = np.concatenate(arrays_to_concat)

    return result
