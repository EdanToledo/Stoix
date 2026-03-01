# flake8: noqa: CCR001
import copy
import os
import queue
import random
import threading
import time
from typing import Callable, Dict, List, Sequence, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from jax import Array
from omegaconf import DictConfig, OmegaConf

from stoix.base_types import (
    ActorApply,
    CoreLearnerState,
    Observation,
    SebulbaExperimentOutput,
    SebulbaLearnerFn,
)
from stoix.evaluator import get_distribution_act_fn, get_sebulba_eval_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.inputs import ArrayInput
from stoix.systems.es.es_types import ESEvaluation
from stoix.systems.es.fitness_shaping import get_fitness_shaping_fn
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvFactory
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.sebulba_utils import (
    AsyncEvaluatorBase,
    OnPolicyPipeline,
    ParameterServer,
    ThreadLifetime,
)
from stoix.utils.timing_utils import TimingTracker
from stoix.utils.training import make_learning_rate

# Memory and performance optimizations for JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"


# ESEvaluation is defined in stoix/systems/es/es_types.py
# Fields: seeds, positive_fitnesses, negative_fitnesses


class AsyncEvaluator(AsyncEvaluatorBase):
    """ES-specific asynchronous evaluator."""

    def run(self) -> None:
        """Run the evaluation loop."""
        while not self.lifetime.should_stop():
            try:
                item = self.eval_queue.get(timeout=1.0)
                if item is None:  # Shutdown signal
                    break

                learner_state, eval_key, eval_step, global_step_count = item
                assert (
                    eval_step == self.eval_step
                ), f"Expected eval_step {self.eval_step}, but got {eval_step}."

                # Prepare parameters for evaluation
                unreplicated_learner_state = unreplicate(learner_state)
                actor_params = jax.block_until_ready(unreplicated_learner_state.params)

                # Run evaluation and log results
                eval_metrics = self.evaluator(actor_params, eval_key)
                self.logger.log(eval_metrics, global_step_count, eval_step, LogEvent.EVAL)

                episode_return = jnp.mean(eval_metrics["episode_return"])

                # Save checkpoint if enabled
                if self.save_checkpoint:
                    self.checkpointer.save(
                        timestep=global_step_count,
                        unreplicated_learner_state=unreplicated_learner_state,
                        episode_return=episode_return,
                    )

                # Track best performance and update progress
                self._update_best_params(episode_return, actor_params)
                self._update_evaluation_progress()
                self.add_eval_metrics(eval_metrics)

            except queue.Empty:
                continue


def get_act_fn(
    apply_fn: ActorApply,
) -> Callable[[FrozenDict, Observation, chex.PRNGKey], Tuple[chex.Array, chex.PRNGKey]]:
    """Create action function for actor threads.

    Note: This function is designed to be vmapped. Each vmapped call receives a single
    observation (no batch dim), so we add/remove a batch dim around the network call.
    """

    def actor_fn(
        params: FrozenDict, observation: Observation, rng_key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.PRNGKey]:
        rng_key, policy_key = jax.random.split(rng_key)
        # Add batch dim for network (vmap strips the env batch dim)
        observation = jnp.expand_dims(observation, axis=0)
        pi = apply_fn(params, observation)
        action = pi.sample(seed=policy_key)
        # Remove batch dim
        action = jnp.squeeze(action, axis=0)
        return action, rng_key

    return actor_fn


def get_rollout_fn(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    parameter_server: "ParameterServer",
    rollout_pipeline: "OnPolicyPipeline",
    apply_fn: ActorApply,
    config: DictConfig,
    logger: StoixLogger,
    learner_devices: Sequence[jax.Device],
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
    unravel_fn: Callable,
    flat_params_shape: Tuple[int, ...],
) -> Callable[[chex.PRNGKey], None]:
    """Create rollout function for actor threads."""

    # Cache frequently used config values
    perturbations_per_actor = config.arch.perturbations_per_actor
    noise_std = config.system.noise_std
    antithetic = config.system.antithetic
    num_episodes_per_eval = config.system.num_episodes_per_eval
    num_actor_threads = config.arch.actor.actor_per_device
    len_actor_device_ids = len(config.arch.actor.device_ids)
    world_size = config.arch.world_size
    actor_log_frequency = config.arch.actor.log_frequency
    num_updates = config.arch.num_updates
    synchronous = config.arch.synchronous

    # Setup action function — vmap over (params, obs, key) for batched perturbation eval
    act_fn = get_act_fn(apply_fn)
    batched_act_fn = jax.jit(jax.vmap(act_fn), device=actor_device)

    @jax.jit
    def prepare_data(evaluation: ESEvaluation) -> ESEvaluation:
        """Prepare and shard evaluation data for learner devices."""
        return ESEvaluation(*jax.tree.map(lambda x: jnp.split(x, len(learner_devices)), evaluation))

    @jax.jit
    def flatten_params(params: FrozenDict) -> chex.Array:
        """Flatten parameter pytree to a single vector."""
        return jax.flatten_util.ravel_pytree(params)[0]

    @jax.jit
    def generate_noise_vectors(perturbation_seeds: chex.Array) -> chex.Array:
        """Generate noise vectors from seeds."""
        return jax.vmap(
            lambda s: jax.random.normal(jax.random.PRNGKey(s), shape=flat_params_shape)
        )(perturbation_seeds)

    @jax.jit
    def compute_perturbed_params(flat_params: chex.Array, all_noise: chex.Array) -> FrozenDict:
        """Compute all perturbed parameter sets for one generation.

        Builds signed noise (antithetic if configured), repeats for multi-episode
        evaluation, and unravels flat vectors back into parameter pytrees.
        """
        if antithetic:
            signed_noise = jnp.concatenate([all_noise, -all_noise], axis=0)
        else:
            signed_noise = all_noise
        if num_episodes_per_eval > 1:
            signed_noise = jnp.repeat(signed_noise, num_episodes_per_eval, axis=0)
        all_perturbed_flat = flat_params + noise_std * signed_noise
        return jax.vmap(unravel_fn)(all_perturbed_flat)

    # Number of parallel policy evaluations per generation
    num_evals = perturbations_per_actor * (2 if antithetic else 1)
    # Total envs = policy evaluations × episodes per evaluation
    num_envs = num_evals * num_episodes_per_eval

    envs = env_factory(num_envs)

    def rollout_fn(rng_key: chex.PRNGKey) -> None:
        """Execute the rollout loop."""
        # Initialize thread state
        thread_start_time = time.perf_counter()
        local_step_count = 0
        actor_policy_version = -1
        num_generations = 0

        # Setup performance tracking
        timer = TimingTracker(maxlen=10)

        with jax.default_device(actor_device):

            while not thread_lifetime.should_stop():

                # Get latest parameters from parameter server
                with timer.time("get_params_time"):
                    # Fetch parameters for the first generation (num_generations=0) to get
                    # initial policy. Skip fetching on the second generation (num_generations=1)
                    # to allow the first learner update to complete. Fetch new parameters
                    # (blocking) for all subsequent generations to get updated policies. This
                    # ensures actors continue evaluating while the learner processes the
                    # previous batch.
                    if not num_generations == 1 or synchronous:
                        params = parameter_server.get_params(thread_lifetime.id)
                        actor_policy_version += 1

                if params is None:  # Shutdown signal
                    break

                # Flatten params
                with timer.time("flatten_params_time"):
                    flat_params = flatten_params(params)

                # Generate perturbation seeds
                rng_key, seed_key = jax.random.split(rng_key)
                perturbation_seeds = jax.random.randint(
                    seed_key,
                    shape=(perturbations_per_actor,),
                    minval=0,
                    maxval=2**31 - 1,
                )

                # Generate noise and compute all perturbed params once per generation
                with timer.time("perturbation_gen_time"):
                    all_noise = generate_noise_vectors(perturbation_seeds)
                    all_params = compute_perturbed_params(flat_params, all_noise)

                # Evaluate all perturbations in parallel
                with timer.time("perturbation_eval_time"):
                    timestep = envs.reset(seed=seeds)
                    episode_returns = np.zeros(num_envs)
                    finished = np.zeros(num_envs, dtype=bool)
                    total_steps = 0

                    # Split RNG keys for each parallel environment
                    rng_key, *eval_keys = jax.random.split(rng_key, num_envs + 1)
                    eval_keys = jnp.stack(eval_keys)

                    while not finished.all():
                        # Batch inference: different params per env
                        with timer.time("inference_time"):
                            actions, eval_keys = batched_act_fn(
                                all_params, timestep.observation, eval_keys
                            )

                        # Move actions to CPU for environment step
                        with timer.time("device_to_host_time"):
                            cpu_actions = np.asarray(actions)

                        # Step all environments
                        with timer.time("env_step_time"):
                            timestep = envs.step(cpu_actions)

                        # Track episode completion using extras metrics
                        with timer.time("episode_tracking_time"):
                            done_np = np.asarray(timestep.last())
                            newly_done = np.logical_and(done_np, ~finished)
                            if newly_done.any():
                                step_returns = np.asarray(
                                    timestep.extras["metrics"]["episode_return"]
                                )
                                episode_returns = np.where(
                                    newly_done, step_returns, episode_returns
                                )
                            finished = np.logical_or(finished, done_np)
                            total_steps += num_envs

                    # End of perturbation evaluation
                    num_generations += 1
                    local_step_count += total_steps

                # Average returns across episodes for each policy evaluation
                # Reshape [num_envs] -> [num_evals, num_episodes_per_eval] and take mean
                eval_returns = episode_returns.reshape(num_evals, num_episodes_per_eval)
                mean_eval_returns = np.mean(eval_returns, axis=1)  # [num_evals]

                # Split results into positive and negative fitnesses
                pos_fitnesses = mean_eval_returns[:perturbations_per_actor]
                if antithetic:
                    neg_fitnesses = mean_eval_returns[perturbations_per_actor:]
                else:
                    neg_fitnesses = np.zeros(perturbations_per_actor)

                # Pack evaluation results
                evaluation = ESEvaluation(
                    seeds=perturbation_seeds,
                    positive_fitnesses=jnp.array(pos_fitnesses),
                    negative_fitnesses=jnp.array(neg_fitnesses),
                )

                # Prepare and send evaluation data to learner
                with timer.time("prepare_data_time"):
                    partitioned_evaluation = prepare_data(evaluation)
                    sharded_evaluation = ESEvaluation(
                        *[
                            jax.device_put_sharded(x, devices=learner_devices)
                            for x in partitioned_evaluation
                        ]
                    )
                    payload = (
                        local_step_count,
                        actor_policy_version,
                        sharded_evaluation,
                    )

                with timer.time("rollout_queue_put_time"):
                    success = rollout_pipeline.send_rollout(thread_lifetime.id, payload)
                    if not success:
                        print(
                            f"Warning: Failed to send evaluation from actor "
                            f"{thread_lifetime.id}"
                        )

                # Periodic logging from primary actor
                if num_generations % actor_log_frequency == 0:
                    if thread_lifetime.id == 0:
                        # Calculate approximate global throughput
                        approximate_global_step = (
                            local_step_count * num_actor_threads * len_actor_device_ids * world_size
                        )

                        # Log timing and throughput metrics
                        timing_metrics = {
                            **timer.get_all_means(),
                            "actor_policy_version": actor_policy_version,
                            "local_env_steps_per_second": int(
                                local_step_count / (time.perf_counter() - thread_start_time)
                            ),
                            "global_env_steps_per_second": int(
                                approximate_global_step / (time.perf_counter() - thread_start_time)
                            ),
                            "num_generations": num_generations,
                            "mean_pos_fitness": float(np.mean(pos_fitnesses)),
                        }
                        logger.log(
                            timing_metrics,
                            approximate_global_step,
                            actor_policy_version,
                            LogEvent.MISC,
                        )

                if num_generations > num_updates:
                    print(
                        f"{Fore.MAGENTA}{Style.BRIGHT}Actor {thread_lifetime.id} has completed "
                        f"{num_generations} generations. Stopping...{Style.RESET_ALL}"
                    )
                    break

            envs.close()

    return rollout_fn


def get_actor_thread(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    parameter_server: ParameterServer,
    rollout_pipeline: OnPolicyPipeline,
    apply_fn: ActorApply,
    rng_key: chex.PRNGKey,
    config: DictConfig,
    seeds: List[int],
    logger: StoixLogger,
    learner_devices: Sequence[jax.Device],
    thread_lifetime: ThreadLifetime,
    unravel_fn: Callable,
    flat_params_shape: Tuple[int, ...],
) -> threading.Thread:
    """Create actor thread for perturbation evaluation."""

    # Ensure RNG key is on correct device
    rng_key = jax.device_put(rng_key, actor_device)

    rollout_fn = get_rollout_fn(
        env_factory=env_factory,
        actor_device=actor_device,
        parameter_server=parameter_server,
        rollout_pipeline=rollout_pipeline,
        apply_fn=apply_fn,
        config=config,
        logger=logger,
        learner_devices=learner_devices,
        seeds=seeds,
        thread_lifetime=thread_lifetime,
        unravel_fn=unravel_fn,
        flat_params_shape=flat_params_shape,
    )

    actor_thread = threading.Thread(
        target=rollout_fn,
        args=(rng_key,),
        name=thread_lifetime.name,
    )

    return actor_thread


def get_learner_step_fn(
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
    flat_params_shape: Tuple[int, ...],
    unravel_fn: Callable,
) -> SebulbaLearnerFn[CoreLearnerState, ESEvaluation]:
    """Create learner update function."""

    # Get ES configuration
    fitness_shaping_fn = get_fitness_shaping_fn(config.system.fitness_shaping)
    noise_std = config.system.noise_std
    antithetic = config.system.antithetic

    def _update_step(
        learner_state: CoreLearnerState,
        sharded_evaluations: List[ESEvaluation],
    ) -> Tuple[CoreLearnerState, Dict[str, Array]]:

        # Combine data from all actors
        evaluation = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *sharded_evaluations)

        params, opt_state, key = learner_state

        # Flatten current parameters
        flat_params = jax.flatten_util.ravel_pytree(params)[0]
        n = evaluation.seeds.shape[0]

        # Regenerate noise vectors from seeds (deterministic via JAX PRNG)
        def _generate_noise(seed: chex.Array) -> chex.Array:
            return jax.random.normal(jax.random.PRNGKey(seed), shape=flat_params_shape)

        noise_vectors = jax.vmap(_generate_noise)(evaluation.seeds)

        # COMPUTE ES GRADIENT
        if antithetic:
            # Apply fitness shaping to all 2n fitness values together
            all_fitnesses = jnp.concatenate(
                [evaluation.positive_fitnesses, evaluation.negative_fitnesses]
            )
            shaped_fitnesses = fitness_shaping_fn(all_fitnesses)
            shaped_pos = shaped_fitnesses[:n]
            shaped_neg = shaped_fitnesses[n:]
            effective_fitnesses = shaped_pos - shaped_neg  # [n]

            # ES gradient: sum(effective_fitness_i * noise_i) / (2 * n * sigma)
            flat_grad = jnp.einsum("i,ij->j", effective_fitnesses, noise_vectors) / (
                2 * n * noise_std
            )
        else:
            # Apply fitness shaping to positive fitnesses only
            shaped_fitnesses = fitness_shaping_fn(evaluation.positive_fitnesses)

            # ES gradient: sum(shaped_fitness_i * noise_i) / (n * sigma)
            flat_grad = jnp.einsum("i,ij->j", shaped_fitnesses, noise_vectors) / (n * noise_std)

        # Negate for maximization (optax minimizes by convention)
        neg_grad_tree = unravel_fn(-flat_grad)

        # Compute the parallel mean (pmean) over learner devices.
        # pmean over devices.
        neg_grad_tree = jax.lax.pmean(neg_grad_tree, axis_name="learner_devices")

        # Update parameters and optimizer state
        updates, new_opt_state = update_fn(neg_grad_tree, opt_state)
        new_params = optax.apply_updates(params, updates)

        # PACK NEW LEARNER STATE
        new_learner_state = CoreLearnerState(new_params, new_opt_state, key)

        # PACK LOSS INFO
        loss_info = {
            "mean_positive_fitness": jnp.mean(evaluation.positive_fitnesses),
            "mean_negative_fitness": jnp.mean(evaluation.negative_fitnesses),
            "grad_norm": jnp.sqrt(jnp.sum(flat_grad**2)),
            "param_norm": jnp.sqrt(jnp.sum(flat_params**2)),
        }

        return new_learner_state, loss_info

    def learner_step_fn(
        learner_state: CoreLearnerState,
        evaluations: List[ESEvaluation],
    ) -> SebulbaExperimentOutput[CoreLearnerState]:
        """Single learner step."""

        learner_state, loss_info = _update_step(learner_state, evaluations)

        return SebulbaExperimentOutput(
            learner_state=learner_state,
            train_metrics=loss_info,
        )

    return learner_step_fn


def get_learner_rollout_fn(
    config: DictConfig,
    parameter_server: ParameterServer,
    rollout_pipeline: OnPolicyPipeline,
    learner_step_fn: Callable,
    logger: StoixLogger,
    async_evaluator: AsyncEvaluator,
) -> Callable[[CoreLearnerState, chex.PRNGKey], None]:
    """Create learner rollout function for network updates."""

    # Cache config values for performance
    learner_log_frequency = config.arch.learner.log_frequency
    num_evaluation = config.arch.num_evaluation
    num_updates_per_eval = config.arch.num_updates_per_eval
    split_key_fn = jax.jit(jax.random.split)

    def learner_rollout(learner_state: CoreLearnerState, rng_key: chex.PRNGKey) -> None:
        """Execute the learner loop."""

        # Initialize learner state
        thread_start_time = time.perf_counter()
        learner_policy_version = 0
        timer = TimingTracker(maxlen=10)

        eval_steps = num_evaluation if num_evaluation > 0 else 1
        # Main training loop organized by evaluation periods
        for eval_step in range(eval_steps):
            for _ in range(num_updates_per_eval):

                # Collect evaluation data from all actors
                with timer.time("rollout_queue_get_time"):
                    rollout_data = rollout_pipeline.collect_rollouts()

                # Process collected data and update step count
                sharded_evaluations = []
                global_step_count = 0
                for local_step_count, _actor_policy_version, sharded_evaluation in rollout_data:
                    global_step_count += local_step_count
                    sharded_evaluations.append(sharded_evaluation)

                # Perform learning update
                with timer.time("learn_step_time"):
                    (learner_state, loss_info,) = learner_step_fn(
                        learner_state,
                        sharded_evaluations,
                    )
                learner_policy_version += 1

                # Broadcast updated parameters to actors
                with timer.time("params_queue_put_time"):
                    parameter_server.distribute_params(learner_state.params)

                # Periodic logging of training progress
                if learner_policy_version % learner_log_frequency == 0:
                    learner_timing_metrics = {
                        **timer.get_all_means(),
                        "global_env_step_count": global_step_count,
                        "learner_policy_version": learner_policy_version,
                        "learner_updates_per_second": int(
                            learner_policy_version / (time.perf_counter() - thread_start_time)
                        ),
                    }
                    logger.log(
                        learner_timing_metrics,
                        global_step_count,
                        learner_policy_version,
                        LogEvent.MISC,
                    )
                    logger.log(
                        loss_info,
                        global_step_count,
                        learner_policy_version,
                        LogEvent.TRAIN,
                    )

            # Submit policy for asynchronous evaluation if evaluation is enabled
            if num_evaluation > 0:
                rng_key, eval_key = split_key_fn(rng_key)
                async_evaluator.submit_evaluation(
                    learner_state, eval_key, eval_step, global_step_count
                )

    return learner_rollout


def get_learner_thread(
    config: DictConfig,
    learn_step: SebulbaLearnerFn[CoreLearnerState, ESEvaluation],
    learner_state: CoreLearnerState,
    parameter_server: ParameterServer,
    rollout_pipeline: OnPolicyPipeline,
    logger: StoixLogger,
    async_evaluator: AsyncEvaluator,
    rng_key: chex.PRNGKey,
) -> threading.Thread:
    """Create learner thread for network updates."""

    learner_rollout_fn = get_learner_rollout_fn(
        config, parameter_server, rollout_pipeline, learn_step, logger, async_evaluator
    )

    learner_thread = threading.Thread(
        target=learner_rollout_fn,
        args=(learner_state, rng_key),
        name="Learner",
    )

    return learner_thread


def stop_all_actor_threads(
    actor_thread_lifetimes: List[ThreadLifetime],
    parameter_server: ParameterServer,
    rollout_pipeline: OnPolicyPipeline,
    actor_threads: List[threading.Thread],
) -> None:
    """Stop all actor threads and clean up resources."""
    # Signal all actors to stop
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Stopping actor threads...{Style.RESET_ALL}")
    for actor_lifetime in actor_thread_lifetimes:
        actor_lifetime.stop()
        print(
            f"{Fore.MAGENTA}{Style.BRIGHT}Actor thread {actor_lifetime.name} "
            f"has stopped.{Style.RESET_ALL}"
        )

    # Clean up communication queues
    parameter_server.clear_all_queues()
    rollout_pipeline.clear_all_queues()
    parameter_server.shutdown_actors()

    # Wait for all threads to complete
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Joining actor threads...{Style.RESET_ALL}")
    for actor_thread in actor_threads:
        actor_thread.join()
        print(
            f"{Fore.MAGENTA}{Style.BRIGHT}Actor thread {actor_thread.name} "
            f"has joined.{Style.RESET_ALL}"
        )


def learner_setup(
    env_factory: EnvFactory,
    keys: chex.Array,
    learner_devices: Sequence[jax.Device],
    config: DictConfig,
) -> Tuple[
    SebulbaLearnerFn[CoreLearnerState, ESEvaluation],
    ActorApply,
    CoreLearnerState,
    Callable,
    Tuple[int, ...],
]:
    """Setup learner networks and initial state."""

    # Get environment specifications
    env = env_factory(num_envs=1)
    num_actions = int(env.action_space().num_values)
    example_obs = env.observation_space().generate_value()
    config.system.action_dim = num_actions
    config.system.observation_shape = example_obs.shape
    env.close()

    key, actor_net_key = keys

    # Build actor network (no critic for ES)
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )

    actor_network = Actor(
        input_layer=ArrayInput(), torso=actor_torso, action_head=actor_action_head
    )

    # Configure learning rate schedule
    lr = make_learning_rate(config.system.lr, config, 1, 1)

    # Setup optimizer with optional gradient clipping
    if config.system.max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.system.max_grad_norm),
            optax.adam(lr, eps=1e-5),
        )
    else:
        optimizer = optax.adam(lr, eps=1e-5)

    # Initialize network parameters
    init_x = example_obs
    init_x = jnp.expand_dims(init_x, axis=0)

    actor_params = actor_network.init(actor_net_key, init_x)
    opt_state = optimizer.init(actor_params)

    # Extract network function and param shape for ES gradient computation
    actor_network_apply_fn = actor_network.apply
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(actor_params)
    flat_params_shape = flat_params.shape

    # Create and compile learner step function
    learn_step = get_learner_step_fn(optimizer.update, config, flat_params_shape, unravel_fn)
    learn_step = jax.pmap(learn_step, axis_name="learner_devices", devices=learner_devices)

    # Load from checkpoint if specified
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=actor_params)
        actor_params = restored_params

    # Initialize complete learner state
    learner_state = CoreLearnerState(actor_params, opt_state, key)

    # Prepare learner state for multi-device training
    learner_state = flax.jax_utils.replicate(learner_state, devices=learner_devices)

    return learn_step, actor_network_apply_fn, learner_state, unravel_fn, flat_params_shape


def check_es_config(config: DictConfig) -> DictConfig:
    """Validate and configure ES-specific parameters.

    Computes perturbations_per_actor, num_envs_per_actor, and num_updates_per_eval
    based on population_size, antithetic sampling, and device configuration.

    Args:
        config: Configuration dictionary containing architecture and system parameters.

    Returns:
        Updated configuration with calculated values.

    Raises:
        AssertionError: If configuration constraints are violated.
    """
    print(f"{Fore.YELLOW}{Style.BRIGHT}Using ES System!{Style.RESET_ALL}")

    total_actors = config.arch.total_num_actor_threads
    population_size = config.system.population_size
    antithetic = config.system.antithetic

    # Compute perturbations per actor
    if antithetic:
        assert population_size % (2 * total_actors) == 0, (
            f"population_size ({population_size}) must be divisible by "
            f"2 * total_num_actor_threads ({2 * total_actors}) when antithetic=True"
        )
        config.arch.perturbations_per_actor = population_size // (2 * total_actors)
    else:
        assert population_size % total_actors == 0, (
            f"population_size ({population_size}) must be divisible by "
            f"total_num_actor_threads ({total_actors})"
        )
        config.arch.perturbations_per_actor = population_size // total_actors

    # Validate sharding constraints
    assert config.arch.perturbations_per_actor % config.num_learner_devices == 0, (
        f"perturbations_per_actor ({config.arch.perturbations_per_actor}) must be "
        f"divisible by num_learner_devices ({config.num_learner_devices})"
    )

    # Compute num_envs_per_actor — one env per (policy evaluation × episode) for max parallelism
    evals_per_actor = config.arch.perturbations_per_actor * (2 if antithetic else 1)
    num_episodes_per_eval = config.system.num_episodes_per_eval
    config.arch.actor.num_envs_per_actor = evals_per_actor * num_episodes_per_eval

    # Compute num_updates_per_eval
    num_updates = config.arch.num_updates
    num_evaluation = max(config.arch.num_evaluation, 1)
    assert num_updates > num_evaluation, (
        f"num_updates ({num_updates}) must be greater than " f"num_evaluation ({num_evaluation})"
    )
    config.arch.num_updates_per_eval = num_updates // num_evaluation

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}ES Config: population_size={population_size}, "
        f"perturbations_per_actor={config.arch.perturbations_per_actor}, "
        f"envs_per_actor={config.arch.actor.num_envs_per_actor}, "
        f"antithetic={antithetic}, num_updates={num_updates}{Style.RESET_ALL}"
    )

    return config


def run_experiment(_config: DictConfig) -> float:
    """Run ES experiment."""
    config = copy.deepcopy(_config)

    # Setup device configuration
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(
        global_devices
    ), "Local and global devices must be the same for now. We dont support multihost just yet"

    # Validate device configurations
    if len(config.arch.actor.device_ids) > len(global_devices):
        raise ValueError(
            f"Number of actor devices ({len(config.arch.actor.device_ids)}) "
            f"is greater than the number of global devices ({len(global_devices)})"
        )
    if len(config.arch.learner.device_ids) > len(global_devices):
        raise ValueError(
            f"Number of learner devices ({len(config.arch.learner.device_ids)}) "
            f"is greater than the number of global devices ({len(global_devices)})"
        )

    # Map device IDs to actual devices
    local_actor_devices = [local_devices[device_id] for device_id in config.arch.actor.device_ids]
    local_learner_devices = [
        local_devices[device_id] for device_id in config.arch.learner.device_ids
    ]
    evaluator_device = local_devices[config.arch.evaluator_device_id]
    world_size = jax.process_count()

    # Log device configuration
    print(f"{Fore.BLUE}{Style.BRIGHT}Actors devices: {local_actor_devices}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}Learner devices: {local_learner_devices}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Global devices: {global_devices}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}Evaluator device: {evaluator_device}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}World size: {world_size}{Style.RESET_ALL}")

    # Update config with computed values
    config.num_learner_devices = len(local_learner_devices)
    config.num_actor_devices = len(local_actor_devices)
    config.arch.world_size = world_size
    config.arch.total_num_actor_threads = (
        len(config.arch.actor.device_ids) * config.arch.actor.actor_per_device
    )

    # Validate and configure ES-specific parameters
    config = check_es_config(config)

    # Setup environment factory
    env_factory = environments.make_factory(config)
    assert isinstance(
        env_factory, EnvFactory
    ), "Environment factory must be an instance of EnvFactory"

    # Initialize random number generators
    key, key_e, actor_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)
    np_rng = np.random.default_rng(config.arch.seed)

    # Setup learner and networks
    learn_step, apply_fn, learner_state, unravel_fn, flat_params_shape = learner_setup(
        env_factory, (key, actor_net_key), local_learner_devices, config
    )

    # Setup evaluation
    eval_act_fn = get_distribution_act_fn(config, apply_fn)
    evaluator, evaluator_envs = get_sebulba_eval_fn(
        env_factory, eval_act_fn, config, np_rng, evaluator_device
    )

    # Setup logging and checkpointing
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,
        )

    # Set global random seeds
    random.seed(config.arch.seed)
    np.random.seed(config.arch.seed)
    key = jax.random.PRNGKey(config.arch.seed)

    # Setup communication infrastructure
    parameter_server = ParameterServer(
        total_num_actors=config.arch.total_num_actor_threads,
        actor_devices=local_actor_devices,
        actors_per_device=config.arch.actor.actor_per_device,
        queue_maxsize=1,
    )

    rollout_pipeline = OnPolicyPipeline(
        total_num_actors=config.arch.total_num_actor_threads, queue_maxsize=1
    )

    # Initialize parameter server with starting parameters
    parameter_server.distribute_params(learner_state.params)

    # Create and start actor threads
    actor_threads = []
    actor_thread_lifetimes = []

    for d_idx, d_id in enumerate(config.arch.actor.device_ids):
        for thread_id in range(config.arch.actor.actor_per_device):
            # Generate unique key and seeds for this actor
            key, thread_key = jax.random.split(key)
            seeds = np_rng.integers(
                np.iinfo(np.int32).max, size=config.arch.actor.num_envs_per_actor
            ).tolist()

            # Create thread with unique identifier
            device_thread_id = d_idx * config.arch.actor.actor_per_device + thread_id
            thread_name = f"Actor-{d_id}-{thread_id}-idx-{device_thread_id}"
            actor_thread_lifetime = ThreadLifetime(
                thread_name=thread_name,
                thread_id=device_thread_id,
            )

            # Create and start actor thread
            actor_thread = get_actor_thread(
                env_factory=env_factory,
                actor_device=local_devices[d_id],
                parameter_server=parameter_server,
                rollout_pipeline=rollout_pipeline,
                apply_fn=apply_fn,
                rng_key=thread_key,
                config=config,
                seeds=seeds,
                logger=logger,
                learner_devices=local_learner_devices,
                thread_lifetime=actor_thread_lifetime,
                unravel_fn=unravel_fn,
                flat_params_shape=flat_params_shape,
            )
            print(
                f"{Fore.BLUE}{Style.BRIGHT}Starting actor thread "
                f"{thread_name}...{Style.RESET_ALL}"
            )
            actor_thread.start()
            actor_threads.append(actor_thread)
            actor_thread_lifetimes.append(actor_thread_lifetime)

    # Setup asynchronous evaluation
    async_eval_lifetime = ThreadLifetime("AsyncEvaluator", 0)
    async_evaluator = AsyncEvaluator(
        evaluator=evaluator,
        logger=logger,
        config=config,
        checkpointer=checkpointer if save_checkpoint else None,
        save_checkpoint=save_checkpoint,
        lifetime=async_eval_lifetime,
    )
    async_evaluator.start()

    # Create and start learner thread
    learner_thread = get_learner_thread(
        config=config,
        learn_step=learn_step,
        learner_state=learner_state,
        parameter_server=parameter_server,
        rollout_pipeline=rollout_pipeline,
        logger=logger,
        async_evaluator=async_evaluator,
        rng_key=key_e,
    )
    start_time = time.perf_counter()
    learner_thread.start()

    # Wait for training to complete
    learner_thread.join()
    end_time = time.perf_counter()
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Learner has finished...{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Learner took {end_time-start_time}.{Style.RESET_ALL}")

    # Cleanup all threads and resources
    stop_all_actor_threads(
        actor_thread_lifetimes, parameter_server, rollout_pipeline, actor_threads
    )
    print(f"{Fore.MAGENTA}{Style.BRIGHT}All actor threads have finished.{Style.RESET_ALL}")

    # Wait for asynchronous evaluations to complete
    async_evaluator.wait_for_all_evaluations()
    # Stop evaluation
    async_eval_lifetime.stop()
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Async evaluator thread has stopped.{Style.RESET_ALL}")
    async_evaluator.join()
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Async evaluator thread has joined.{Style.RESET_ALL}")

    # Measure absolute metric.
    if config.arch.absolute_metric:
        best_params = async_evaluator.get_best_params()
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        abs_metric_evaluator, abs_metric_evaluator_envs = get_sebulba_eval_fn(
            env_factory, eval_act_fn, config, np_rng, evaluator_device, eval_multiplier=10
        )
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = abs_metric_evaluator(best_params, eval_key)

        total_ts = (
            config.arch.total_timesteps if config.arch.total_timesteps else config.arch.num_updates
        )
        logger.log(
            eval_metrics,
            int(total_ts),
            int(config.arch.num_evaluation - 1),
            LogEvent.ABSOLUTE,
        )
        abs_metric_evaluator_envs.close()

        # Use the absolute metric evaluation for final performance
        eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))
    else:
        # Use the last evaluation from the async evaluator
        eval_performance = float(async_evaluator.get_final_episode_return())

    # Final cleanup
    evaluator_envs.close()
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Evaluator environments have been closed.{Style.RESET_ALL}")
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/sebulba",
    config_name="default_ff_es.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    start = time.perf_counter()
    eval_performance = run_experiment(cfg)
    end = time.perf_counter()
    print(
        f"{Fore.CYAN}{Style.BRIGHT}ES experiment completed in "
        f"{end - start:.2f}s with a final episode return of {eval_performance}.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
