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
    """Handles rollout communication for on-policy Sebulba systems."""

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
    """Handles parameter distribution for Sebulba systems."""

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
        self.param_queues = self._create_param_queues(total_num_actors, queue_maxsize)

    def _create_param_queues(self, total_num_actors: int, queue_maxsize: int) -> List[queue.Queue]:
        """Create dedicated queues for each actor's parameters."""
        return [queue.Queue(maxsize=queue_maxsize) for _ in range(total_num_actors)]

    def _unreplicate_params(self, params: Parameters) -> Optional[Parameters]:
        """Convert from replicated to single copy for distribution."""
        try:
            return flax.jax_utils.unreplicate(params)
        except Exception as e:
            warnings.warn(f"Failed to unreplicate parameters: {e}", stacklevel=2)
            return None

    def distribute_params(
        self,
        params: Parameters,
        block: bool = True,
        timeout: Optional[float] = None,
        block_params_until_ready: bool = False,
    ) -> None:
        """Distribute parameters to all actors with device placement."""
        unreplicated_params = self._unreplicate_params(params)
        if unreplicated_params is None:
            return

        self._distribute_to_devices(unreplicated_params, block, timeout, block_params_until_ready)

    def _distribute_to_devices(
        self,
        unreplicated_params: Parameters,
        block: bool,
        timeout: Optional[float],
        block_params_until_ready: bool,
    ) -> None:
        """Distribute parameters across all devices."""
        actor_idx = 0
        for device in self.actor_devices:
            device_params = self._prepare_device_params(
                unreplicated_params, device, block_params_until_ready
            )

            if device_params is not None:
                self._distribute_to_device_actors(device_params, actor_idx, block, timeout)

            actor_idx += self.actors_per_device

    def _prepare_device_params(
        self,
        params: Parameters,
        device: jax.Device,
        block_until_ready: bool,
    ) -> Optional[Parameters]:
        """Prepare parameters for a specific device."""
        try:
            device_params = jax.device_put(params, device)
            if block_until_ready:
                device_params = jax.block_until_ready(device_params)
            return device_params
        except Exception as e:
            warnings.warn(f"Failed to place parameters on device {device}: {e}", stacklevel=2)
            return None

    def _distribute_to_device_actors(
        self,
        device_params: Parameters,
        start_actor_idx: int,
        block: bool,
        timeout: Optional[float],
    ) -> None:
        """Distribute parameters to all actors on a device."""
        for i in range(self.actors_per_device):
            actor_idx = start_actor_idx + i
            self._put_params_in_queue(actor_idx, device_params, block, timeout)

    def _put_params_in_queue(
        self,
        actor_idx: int,
        params: Parameters,
        block: bool,
        timeout: Optional[float],
    ) -> None:
        """Put parameters in an actor's queue."""
        try:
            if block:
                self._blocking_put(actor_idx, params, timeout)
            else:
                self.param_queues[actor_idx].put_nowait(params)
        except (queue.Full, queue.Empty):
            warnings.warn(f"Failed to put parameters in queue {actor_idx}", stacklevel=2)

    def _blocking_put(
        self,
        actor_idx: int,
        params: Parameters,
        timeout: Optional[float],
    ) -> None:
        """Perform blocking put operation with optional timeout."""
        if timeout is not None:
            self.param_queues[actor_idx].put(params, timeout=timeout)
        else:
            self.param_queues[actor_idx].put(params)

    def get_params(self, actor_idx: int, timeout: Optional[float] = None) -> Optional[Parameters]:
        """Get parameters for an actor."""
        params = self._get_from_queue(actor_idx, timeout)
        if params is None:
            return None
        return jax.block_until_ready(params)

    def _get_from_queue(
        self,
        actor_idx: int,
        timeout: Optional[float],
    ) -> Optional[Parameters]:
        """Get parameters from an actor's queue."""
        try:
            if timeout is not None:
                return self.param_queues[actor_idx].get(timeout=timeout)
            else:
                return self.param_queues[actor_idx].get()
        except queue.Empty:
            return None

    def shutdown_actors(self) -> None:
        """Send shutdown signals to all actors."""
        for param_queue in self.param_queues:
            self._send_shutdown_signal(param_queue)

    def _send_shutdown_signal(self, param_queue: queue.Queue) -> None:
        """Send shutdown signal to a single queue."""
        try:
            param_queue.put_nowait(None)
        except queue.Full:
            pass  # Actor will eventually check lifetime

    def clear_all_queues(self) -> None:
        """Clear all parameter queues."""
        for param_queue in self.param_queues:
            self._clear_queue(param_queue)

    def _clear_queue(self, param_queue: queue.Queue) -> None:
        """Clear a single queue."""
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
        self.max_episode_return = -jnp.inf
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
