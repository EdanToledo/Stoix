import copy
import queue
import threading
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from colorama import Fore, Style
from flax.jax_utils import unreplicate
from jumanji.types import TimeStep
from omegaconf import DictConfig

from stoix.base_types import Parameters, StoixTransition
from stoix.utils.logger import StoixLogger

DEFAULT_TIMEOUT = 180
QUEUE_POLL_TIMEOUT = 1.0


class OperationResult(Enum):
    """Enum for pipeline operation results to avoid string-based error handling."""
    SUCCESS = "success"
    QUEUE_FULL = "queue_full"
    SHUTDOWN = "shutdown"


class ThreadLifetime:
    """Simple class for a mutable boolean that can be used to signal a thread to stop."""

    def __init__(self) -> None:
        self._stop = False

    def should_stop(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True


class QueueStats:
    """Encapsulates queue monitoring statistics."""
    
    def __init__(self) -> None:
        self.peak_size = 0
        self.full_count = 0
    
    def update_peak(self, current_size: int) -> None:
        """Update peak size if current size is larger."""
        self.peak_size = max(self.peak_size, current_size)
    
    def increment_full_count(self) -> None:
        """Increment the full event counter."""
        self.full_count += 1
    
    def reset(self, current_size: int = 0) -> None:
        """Reset statistics for a new measurement period."""
        self.peak_size = current_size
        self.full_count = 0
    
    def to_dict(self, current_size: int) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {
            "queue_current_size": current_size,
            "queue_peak_size": self.peak_size,
            "queue_full_events_count": self.full_count,
        }


class OnPolicyPipeline(threading.Thread):
    """
    The Pipeline shards trajectories into learner_devices,
    ensuring trajectories are consumed in the right order to avoid being off-policy
    and limit the max number of samples in device memory at one time to avoid OOM issues.
    """

    def __init__(self, max_size: int, learner_devices: List[jax.Device], lifetime: ThreadLifetime):
        """
        Initialize the pipeline with a maximum size and the devices to shard trajectories across.

        Args:
            max_size: The maximum number of trajectories to keep in the pipeline.
            learner_devices: The devices to shard trajectories across.
            lifetime: Thread lifetime manager for coordinated shutdown.
        """
        super().__init__(name="Pipeline")
        self.learner_devices = learner_devices
        self.sharding_queue: queue.Queue = queue.Queue()
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self.lifetime = lifetime
        self._max_size = max_size
        self._stats = QueueStats()
        
        # Pre-compile JAX operations for performance
        self._split_axis = jax.jit(
            lambda x, y: jnp.split(x, len(self.learner_devices), axis=y), 
            static_argnums=(1,)
        )
        self._stack_trajectory = jax.jit(
            lambda *trajectory: jax.tree_map(lambda *x: jnp.stack(x, axis=0), *trajectory)
        )
        self._concat_metrics_fn = jax.jit(
            lambda *actor_metrics: jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *actor_metrics)
        )

    def _shard_data(self, payload: Any, axis: int) -> Any:
        """Shard data across learner devices. Device operations cannot be JIT-compiled."""
        split_payload = self._split_axis(payload, axis)
        return jax.device_put_sharded(split_payload, devices=self.learner_devices)

    def _process_sharding_request(self, item: Tuple) -> OperationResult:
        """Process a single sharding request and put result in main queue."""
        traj, timestep, actor_timings_dict, actor_episode_metrics, result_queue = item
        
        try:
            # Perform sharding operations
            sharded_traj = jax.tree.map(lambda x: self._shard_data(x, 1), traj)
            sharded_timestep = jax.tree.map(lambda x: self._shard_data(x, 0), timestep)
            
            current_size = self._queue.qsize()
            
            # Attempt to put the result in the main queue
            self._queue.put(
                (sharded_traj, sharded_timestep, actor_timings_dict, actor_episode_metrics),
                block=True,
                timeout=DEFAULT_TIMEOUT,
            )
            
            self._stats.update_peak(self._queue.qsize())
            result_queue.put(OperationResult.SUCCESS)
            return OperationResult.SUCCESS
            
        except queue.Full:
            self._stats.increment_full_count()
            self._stats.update_peak(current_size)
            result_queue.put(OperationResult.QUEUE_FULL)
            self._log_queue_full_warning(current_size)
            return OperationResult.QUEUE_FULL
            
        except Exception as e:
            result_queue.put(e)
            return OperationResult.QUEUE_FULL  # Treat as failure

    def _log_queue_full_warning(self, current_size: int) -> None:
        """Log a warning when the queue becomes full."""
        print(
            f"{Fore.RED}{Style.BRIGHT}Pipeline is full and actor has timed out. "
            f"Queue size: {current_size}/{self._max_size}, Peak: {self._stats.peak_size}, "
            f"Full events: {self._stats.full_count}. A deadlock might be occurring{Style.RESET_ALL}"
        )

    def run(self) -> None:
        """Handle sharding operations in a dedicated thread to avoid blocking actors."""
        while not self.lifetime.should_stop():
            try:
                item = self.sharding_queue.get(timeout=QUEUE_POLL_TIMEOUT)
                if item is None:  # Sentinel for shutdown
                    break
                
                self._process_sharding_request(item)
                    
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
        if not self.is_alive():
            raise RuntimeError("Pipeline thread has died")
        
        # Pre-process data (each actor does this independently)
        # [Transition(num_envs)] * rollout_len --> Transition[(rollout_len, num_envs,)
        traj = self._stack_trajectory(*traj)
        # List[Dict[str, List[float]]] --> Dict[str, List[float]]
        actor_episode_metrics = self._concat_metrics_fn(*actor_episode_metrics)
        
        # Create result queue and send to sharding thread
        result_queue = queue.Queue(maxsize=1)
        self.sharding_queue.put((traj, timestep, actor_timings_dict, actor_episode_metrics, result_queue))
        
        # Wait for result and handle errors
        result = self._wait_for_sharding_result(result_queue)
        self._handle_sharding_result(result)

    def _wait_for_sharding_result(self, result_queue: queue.Queue) -> Any:
        """Wait for sharding operation to complete and return result."""
        try:
            return result_queue.get(timeout=DEFAULT_TIMEOUT)
        except queue.Empty:
            raise TimeoutError("Sharding operation timed out")

    def _handle_sharding_result(self, result: Any) -> None:
        """Handle the result from a sharding operation."""
        if result == OperationResult.QUEUE_FULL:
            raise queue.Full("Pipeline queue is full")
        elif isinstance(result, Exception):
            raise result
        elif result != OperationResult.SUCCESS:
            raise RuntimeError(f"Unexpected pipeline error: {result}")

    def qsize(self) -> int:
        """Return the number of trajectories in the pipeline."""
        return self._queue.qsize()

    def get_queue_stats(self) -> Dict[str, int]:
        """Get comprehensive queue statistics."""
        return self._stats.to_dict(self._queue.qsize())

    def reset_stats(self) -> None:
        """Reset queue statistics for a new measurement period."""
        self._stats.reset(self._queue.qsize())

    def get(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[StoixTransition, TimeStep, Dict[str, List[float]], Dict[str, jnp.ndarray]]:
        """Get a trajectory from the pipeline."""
        return self._queue.get(block, timeout)  # type: ignore

    def _clear_queue(self, target_queue: queue.Queue) -> None:
        """Clear all items from a queue."""
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break

    def clear(self) -> None:
        """Clear the pipeline."""
        self._clear_queue(self.sharding_queue)
        self._clear_queue(self._queue)

    def shutdown(self) -> None:
        """Gracefully shutdown the pipeline."""
        try:
            self.sharding_queue.put(None)  # Send sentinel
        except:
            pass  # Queue might be closed or full
        self.clear()


class ParamsSource(threading.Thread):
    """
    A ParamSource allows network params to be passed from a
    Learner component to Actor components.
    """

    def __init__(self, init_value: Parameters, device: jax.Device, lifetime: ThreadLifetime):
        super().__init__(name=f"ParamsSource-{device.id}")
        self.value: Parameters = jax.device_put(init_value, device)
        self.device = device
        self.new_value: queue.Queue = queue.Queue(maxsize=1)
        self.lifetime = lifetime

    def run(self) -> None:
        """Update the ParamSource value when new parameters are available."""
        while not self.lifetime.should_stop():
            try:
                new_params = self.new_value.get(block=True, timeout=QUEUE_POLL_TIMEOUT)
                new_params = jax.block_until_ready(new_params)
                self.value = jax.device_put(new_params, self.device)
            except queue.Empty:
                continue

    def update(self, new_params: Parameters) -> None:
        """
        Update the ParamSource with new parameters.

        Args:
            new_params: The new parameters to update with.
        """
        # Keep only the latest update - remove old one if queue is full
        if self.new_value.full():
            try:
                self.new_value.get_nowait()
            except queue.Empty:
                pass
        
        # Put new params (use put_nowait to avoid blocking)
        try:
            self.new_value.put_nowait(new_params)
        except queue.Full:
            # Defensive handling - should not happen since we cleared above
            pass

    def get(self) -> Parameters:
        """Get the current parameters."""
        return self.value


class RecordTimeTo:
    """Context manager for recording execution time to a list."""
    
    def __init__(self, to: List[float]):
        self.to = to

    def __enter__(self) -> None:
        self.start = time.monotonic()

    def __exit__(self, *args: Any) -> None:
        end = time.monotonic()
        self.to.append(end - self.start)


class AsyncEvaluatorBase(threading.Thread, ABC):
    """Abstract base class for asynchronous evaluators that run evaluation without blocking the learner."""
    
    def __init__(
        self,
        evaluator: Callable,
        logger: StoixLogger,
        config: DictConfig,
        checkpointer: Any,
        save_checkpoint: bool,
        steps_consumed_per_eval: int,
        lifetime: "ThreadLifetime",
    ):
        super().__init__(name="AsyncEvaluator")
        self.evaluator = evaluator
        self.logger = logger
        self.config = config
        self.checkpointer = checkpointer
        self.save_checkpoint = save_checkpoint
        self.steps_consumed_per_eval = steps_consumed_per_eval
        self.lifetime = lifetime
        self.eval_queue: queue.Queue = queue.Queue()
        self.max_episode_return = jnp.float32(-1e7)
        self.best_params = None
        self.eval_step = 0
        self.expected_evaluations = config.arch.num_evaluation
        self.completed_evaluations = 0
        self._evaluation_lock = threading.Lock()
        self._all_evaluations_done = threading.Event()
        
    @abstractmethod
    def run(self) -> None:
        """Run the asynchronous evaluation loop. Must be implemented by subclasses."""
        pass
        
    def submit_evaluation(
        self, 
        episode_metrics: Any, 
        train_metrics: Any, 
        learner_state: Any, 
        timings_dict: Any,
        eval_key: chex.PRNGKey,
        eval_step: int
    ) -> None:
        """Submit evaluation data to be processed asynchronously."""
        try:
            self.eval_queue.put_nowait((episode_metrics, train_metrics, learner_state, timings_dict, eval_key))
            print(f"{Fore.YELLOW}{Style.BRIGHT}Submitted evaluation {eval_step+1}/{self.expected_evaluations}, Current Eval Queue Size: {self.eval_queue.qsize()}{Style.RESET_ALL}")
        except queue.Full:
            # If queue is full, skip this evaluation to avoid blocking
            warnings.warn("Evaluation queue is full, skipping evaluation", stacklevel=2)
            
    def get_best_params(self) -> Any:
        """Get the best parameters found so far."""
        return self.best_params
        
    def wait_for_all_evaluations(self, timeout: float = 300.0) -> bool:
        """Wait for all evaluations to complete. Returns True if completed, False if timed out."""
        return self._all_evaluations_done.wait(timeout)
        
    def shutdown(self) -> None:
        """Gracefully shutdown the evaluator."""
        try:
            self.eval_queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass

    def _update_evaluation_progress(self) -> None:
        """Update evaluation progress and signal completion if all evaluations are done."""
        self.eval_step += 1
        
        # Check if all evaluations are completed
        with self._evaluation_lock:
            self.completed_evaluations += 1
            if self.completed_evaluations >= self.expected_evaluations:
                self._all_evaluations_done.set()

    def _extract_actor_params(self, learner_state: Any) -> Any:
        """Extract actor parameters from learner state. Override if needed for specific algorithms."""
        return unreplicate(learner_state.params.actor_params)

    def _calculate_timestep(self) -> int:
        """Calculate the current timestep for logging."""
        return int(self.steps_consumed_per_eval * (self.eval_step + 1))

    def _update_best_params(self, episode_return: Any, actor_params: Any) -> None:
        """Update the best parameters if the current episode return is better."""
        if self.config.arch.absolute_metric and self.max_episode_return <= episode_return:
            self.best_params = copy.deepcopy(actor_params)
            self.max_episode_return = episode_return
