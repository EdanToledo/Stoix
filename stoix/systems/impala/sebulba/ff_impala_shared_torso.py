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
import rlax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from jax import Array
from omegaconf import DictConfig, OmegaConf
from stoa.core_wrappers.episode_metrics import get_final_step_metrics

from stoix.base_types import (
    Action,
    ActorApply,
    ActorCriticApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CoreLearnerState,
    CriticApply,
    LogProb,
    Observation,
    SebulbaExperimentOutput,
    SebulbaLearnerFn,
)
from stoix.evaluator import get_distribution_act_fn, get_sebulba_eval_fn
from stoix.networks.base import FeedForwardActorCritic as ActorCritic
from stoix.networks.inputs import ArrayInput
from stoix.systems.impala.impala_types import ImpalaTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvFactory
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.sebulba_utils import (
    AsyncEvaluatorBase,
    OnPolicyPipeline,
    ParameterServer,
    ThreadLifetime,
    tree_stack_numpy,
)
from stoix.utils.timing_utils import TimingTracker
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate

# Memory and performance optimizations for JAX
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


class AsyncEvaluator(AsyncEvaluatorBase):
    """IMPALA-specific asynchronous evaluator."""

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
    actor_critic_apply_fn: ActorCriticApply,
) -> Callable[[FrozenDict, Observation, chex.PRNGKey], Tuple[Action, LogProb, chex.PRNGKey],]:
    """Create action function for actor threads."""

    def actor_fn(
        params: FrozenDict, observation: Observation, rng_key: chex.PRNGKey
    ) -> Tuple[Action, LogProb, chex.PRNGKey]:
        rng_key, policy_key = jax.random.split(rng_key)
        pi, _ = actor_critic_apply_fn(params, observation)
        action = pi.sample(seed=policy_key)
        log_prob = pi.log_prob(action)
        return action, log_prob, rng_key

    return actor_fn


def get_rollout_fn(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    parameter_server: ParameterServer,
    rollout_pipeline: OnPolicyPipeline,
    actor_critic_apply_fn: ActorCriticApply,
    config: DictConfig,
    logger: StoixLogger,
    learner_devices: Sequence[jax.Device],
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
) -> Callable[[chex.PRNGKey], None]:
    """Create rollout function for actor threads."""

    # Setup action function and data preparation
    act_fn = get_act_fn(actor_critic_apply_fn)
    act_fn = jax.jit(act_fn, device=actor_device)

    @jax.jit
    def prepare_data(storage: List[ImpalaTransition]) -> List[ImpalaTransition]:
        """Prepare and shard trajectory data for learner devices."""
        split_data: List[ImpalaTransition] = jax.tree.map(
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )
        return split_data

    # Cache frequently used config values
    num_envs_per_actor = config.arch.actor.num_envs_per_actor
    rollout_length = config.system.rollout_length
    num_actor_threads = config.arch.actor.actor_per_device
    len_actor_device_ids = len(config.arch.actor.device_ids)
    world_size = config.arch.world_size
    actor_log_frequency = config.arch.actor.log_frequency
    num_updates = config.arch.num_updates
    synchronous = config.arch.synchronous

    envs = env_factory(num_envs_per_actor)

    def rollout_fn(rng_key: chex.PRNGKey) -> None:
        """Execute the rollout loop."""
        # Initialize thread state
        thread_start_time = time.perf_counter()
        local_step_count = 0
        actor_policy_version = -1
        num_rollouts = 0

        # Setup performance tracking and storage
        timer = TimingTracker(maxlen=10)
        traj_storage: List[ImpalaTransition] = []
        episode_metrics_storage = []

        with jax.default_device(actor_device):
            timestep = envs.reset(seed=seeds)

            while not thread_lifetime.should_stop():

                # Calculate rollout length (include bootstrap step for first rollout)
                num_steps_with_bootstrap = rollout_length + int(len(traj_storage) == 0)

                # Get latest parameters from parameter server
                with timer.time("get_params_time"):
                    # Fetch parameters for the first rollout (num_rollouts=0) to get initial policy
                    # Skip fetching on the second rollout (num_rollouts=1) to allow the first learner
                    # update to complete. Fetch new parameters (blocking) for all subsequent rollouts
                    # to get updated policies. This ensures actors continue collecting data while the
                    # learner processes the previous batch.
                    if not num_rollouts == 1 or synchronous:
                        params = parameter_server.get_params(thread_lifetime.id)
                        actor_policy_version += 1

                if params is None:  # Shutdown signal
                    break

                # Collect trajectory data
                with timer.time("single_actor_rollout_time"):
                    for _ in range(num_steps_with_bootstrap):
                        obs_tm1 = timestep.observation

                        # Get action from policy
                        with timer.time("inference_time"):
                            a_tm1, log_prob_tm1, rng_key = act_fn(params, obs_tm1, rng_key)

                        # Move action to CPU for environment step
                        with timer.time("device_to_host_time"):
                            cpu_action = np.asarray(a_tm1)

                        # Step environment
                        with timer.time("env_step_time"):
                            timestep = envs.step(cpu_action)

                        # Store transition data
                        with timer.time("storage_time"):
                            reward_t = timestep.reward
                            done_t = timestep.last()
                            trunc_t = jnp.logical_and(done_t, timestep.discount == 1)
                            timestep_metrics = timestep.extras["metrics"]
                            traj_storage.append(
                                ImpalaTransition(
                                    obs=obs_tm1,
                                    done=done_t,
                                    truncated=trunc_t,
                                    action=a_tm1,
                                    log_prob=log_prob_tm1,
                                    reward=reward_t,
                                )
                            )
                            episode_metrics_storage.append(timestep_metrics)
                        local_step_count += len(done_t)
                    # End of rollout collection
                    num_rollouts += 1
                # Prepare and send rollout data to learner
                with timer.time("prepare_data_time"):
                    partitioned_traj_storage = prepare_data(traj_storage)
                    sharded_traj_storage = ImpalaTransition(
                        *[
                            jax.device_put_sharded(x, devices=learner_devices)
                            for x in partitioned_traj_storage
                        ]
                    )
                    payload = (
                        local_step_count,
                        actor_policy_version,
                        sharded_traj_storage,
                    )

                with timer.time("rollout_queue_put_time"):
                    success = rollout_pipeline.send_rollout(thread_lifetime.id, payload)
                    if not success:
                        print(f"Warning: Failed to send rollout from actor {thread_lifetime.id}")

                # Keep last transition for next rollout's bootstrap
                traj_storage = traj_storage[-1:]

                # Periodic logging from primary actor
                if num_rollouts % actor_log_frequency == 0:
                    if thread_lifetime.id == 0:
                        # Calculate approximate global throughput
                        approximate_global_step = (
                            local_step_count * num_actor_threads * len_actor_device_ids * world_size
                        )

                        # Log timing and throughput metrics
                        timing_metrics = {
                            **timer.get_all_means(),
                            "actor_policy_version": actor_policy_version,
                            "local_SPS": int(
                                local_step_count / (time.perf_counter() - thread_start_time)
                            ),
                            "global_SPS": int(
                                approximate_global_step / (time.perf_counter() - thread_start_time)
                            ),
                            "num_rollouts": num_rollouts,
                        }
                        logger.log(
                            timing_metrics,
                            approximate_global_step,
                            actor_policy_version,
                            LogEvent.MISC,
                        )

                        # Log episode metrics if any episodes completed
                        concat_episode_metrics = tree_stack_numpy(episode_metrics_storage)
                        actor_metrics, has_final_ep_step = get_final_step_metrics(
                            concat_episode_metrics
                        )
                        if has_final_ep_step:
                            actor_metrics["num_completed_episodes_in_rollout_batch"] = len(
                                actor_metrics["episode_return"]
                            )
                            if actor_metrics["num_completed_episodes_in_rollout_batch"] > 1:
                                logger.log(
                                    actor_metrics,
                                    approximate_global_step,
                                    actor_policy_version,
                                    LogEvent.ACT,
                                )
                                episode_metrics_storage.clear()

                if num_rollouts > num_updates:
                    print(
                        f"{Fore.MAGENTA}{Style.BRIGHT}Actor {thread_lifetime.id} has completed "
                        f"{num_rollouts} rollouts. Stopping...{Style.RESET_ALL}"
                    )
                    break

            envs.close()

    return rollout_fn


def get_actor_thread(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    parameter_server: "ParameterServer",
    rollout_pipeline: "OnPolicyPipeline",
    actor_critic_apply_fn: ActorCriticApply,
    rng_key: chex.PRNGKey,
    config: DictConfig,
    seeds: List[int],
    logger: StoixLogger,
    learner_devices: Sequence[jax.Device],
    thread_lifetime: ThreadLifetime,
) -> threading.Thread:
    """Create actor thread for environment data collection."""

    # Ensure RNG key is on correct device
    rng_key = jax.device_put(rng_key, actor_device)

    rollout_fn = get_rollout_fn(
        env_factory=env_factory,
        actor_device=actor_device,
        parameter_server=parameter_server,
        rollout_pipeline=rollout_pipeline,
        actor_critic_apply_fn=actor_critic_apply_fn,
        config=config,
        logger=logger,
        learner_devices=learner_devices,
        seeds=seeds,
        thread_lifetime=thread_lifetime,
    )

    actor_thread = threading.Thread(
        target=rollout_fn,
        args=(rng_key,),
        name=thread_lifetime.name,
    )

    return actor_thread


def get_learner_step_fn(
    actor_critic_apply_fn: ActorCriticApply,
    actor_critic_update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> SebulbaLearnerFn[CoreLearnerState, ImpalaTransition]:
    """Create learner update function using V-trace."""

    def _update_step(
        learner_state: CoreLearnerState, sharded_traj_batchs: List[ImpalaTransition]
    ) -> Tuple[CoreLearnerState, Dict[str, Array]]:
        """Single V-trace update step."""
        params, opt_states, key = learner_state

        # Combine data from all actors
        traj_batch = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_traj_batchs)

        # Extract trajectory components
        o_tm1 = traj_batch.obs
        a_tm1 = traj_batch.action
        behavior_log_prob_tm1 = traj_batch.log_prob
        r_t = traj_batch.reward
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)

        # Optional reward normalization
        if config.system.normalize_rewards:
            r_mean = jnp.mean(r_t)
            r_std = jnp.std(r_t)
            r_t = config.system.reward_scale * (r_t - r_mean) / (r_std + config.system.reward_eps)

        # Track reward statistics for monitoring
        extra_metrics = {
            "reward_mean": r_mean if config.system.normalize_rewards else jnp.mean(r_t),
            "reward_std": r_std if config.system.normalize_rewards else jnp.std(r_t),
        }

        def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
            """Update networks for single minibatch."""

            params, opt_states = train_state
            (
                o_tm1_batch,
                a_tm1_batch,
                r_t_batch,
                d_t_batch,
                behavior_log_prob_tm1_batch,
            ) = batch_info

            def _actor_critic_loss_fn(
                actor_critic_params: FrozenDict,
                o_tm1: Array,
                a_tm1: Array,
                behavior_log_prob_tm1: Array,
                r_t: Array,
                d_t: Array,
            ) -> Tuple[Array, Dict[str, Array]]:
                """Combined actor-critic loss using V-trace targets."""

                # Compute importance sampling ratios with current policy
                pi_tm1, v_tm1 = actor_critic_apply_fn(actor_critic_params, o_tm1)
                log_prob_tm1 = pi_tm1.log_prob(a_tm1)
                rho_tm1 = jax.lax.stop_gradient(jnp.exp(log_prob_tm1 - behavior_log_prob_tm1))

                # Due to bootstrapping, we need to shorten the sequences by 1
                v_t = v_tm1[1:]
                v_tm1 = v_tm1[:-1]
                r_t = r_t[:-1]
                d_t = d_t[:-1]
                rho_tm1 = rho_tm1[:-1]
                log_prob_tm1 = log_prob_tm1[:-1]

                # Apply V-trace algorithm for off-policy correction
                vtrace_outputs = jax.vmap(
                    rlax.vtrace_td_error_and_advantage,
                    in_axes=(1, 1, 1, 1, 1, None, None, None),
                    out_axes=1,
                )(
                    v_tm1,
                    v_t,
                    r_t,
                    d_t,
                    rho_tm1,
                    config.system.vtrace_lambda,
                    config.system.clip_rho_threshold,
                    config.system.clip_pg_rho_threshold,
                )

                # Critic loss
                value_loss = 0.5 * jnp.sum(jnp.square(vtrace_outputs.errors))
                critic_loss = config.system.vf_coef * value_loss

                # Actor loss using V-trace advantages
                policy_loss = -(
                    jax.lax.stop_gradient(vtrace_outputs.pg_advantage) * log_prob_tm1
                ).sum()

                # Entropy regularization for exploration
                entropy = pi_tm1.entropy().sum()
                actor_loss = policy_loss - config.system.ent_coef * entropy

                # Combined total loss
                total_loss = critic_loss + actor_loss

                # Collect all loss info
                loss_info = {
                    "total_loss": total_loss,
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss,
                    "entropy": entropy,
                    "q_estimate": vtrace_outputs.q_estimate.mean(),
                    "pg_advantage": vtrace_outputs.pg_advantage.mean(),
                }

                return total_loss, loss_info

            # Compute gradients
            actor_critic_grad_fn = jax.grad(_actor_critic_loss_fn, has_aux=True)
            actor_critic_grads, loss_info = actor_critic_grad_fn(
                params,
                o_tm1_batch,
                a_tm1_batch,
                behavior_log_prob_tm1_batch,
                r_t_batch,
                d_t_batch,
            )

            # Synchronize gradients across devices
            actor_critic_grads, loss_info = jax.lax.pmean(
                (actor_critic_grads, loss_info),
                axis_name="learner_devices",
            )

            # Apply parameter updates
            actor_critic_updates, actor_critic_new_opt_state = actor_critic_update_fn(
                actor_critic_grads, opt_states
            )
            actor_critic_new_params = optax.apply_updates(params, actor_critic_updates)

            return (actor_critic_new_params, actor_critic_new_opt_state), loss_info

        # Prepare data for minibatch processing
        envs_per_batch = config.arch.total_num_envs // len(config.arch.learner.device_ids)

        batch = (o_tm1, a_tm1, r_t, d_t, behavior_log_prob_tm1)
        chex.assert_tree_shape_prefix(
            (a_tm1, r_t, d_t, behavior_log_prob_tm1),
            (
                config.system.rollout_length + 1,
                envs_per_batch,
            ),
        )

        # Split data across minibatches for memory efficiency
        envs_per_minibatch = envs_per_batch // config.system.num_minibatches
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(
                x, [-1, config.system.num_minibatches, envs_per_minibatch] + list(x.shape[2:])
            ),
            batch,
        )
        # Transpose to have minibatches as leading dimension
        minibatches = jax.tree.map(
            lambda x: jnp.transpose(x, [1, 0, 2] + list(range(3, x.ndim))),
            minibatches,
        )
        chex.assert_tree_shape_prefix(
            minibatches[1:],
            (
                config.system.num_minibatches,
                config.system.rollout_length + 1,
                envs_per_minibatch,
            ),
        )

        # Process all minibatches sequentially
        (params, opt_states), loss_info = jax.lax.scan(
            _update_minibatch, (params, opt_states), minibatches
        )

        # Combine loss info with reward statistics
        loss_info = {**loss_info, **extra_metrics}
        learner_state = CoreLearnerState(params, opt_states, key)
        return learner_state, loss_info

    def learner_step_fn(
        learner_state: CoreLearnerState, traj_batch: List[ImpalaTransition]
    ) -> SebulbaExperimentOutput[CoreLearnerState]:
        """Single learner step."""
        learner_state, loss_info = _update_step(learner_state, traj_batch)

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

                # Collect rollout data from all actors
                with timer.time("rollout_queue_get_time"):
                    rollout_data = rollout_pipeline.collect_rollouts()

                # Process collected data and update step count
                sharded_storages = []
                global_step_count = 0
                for local_step_count, _actor_policy_version, sharded_storage in rollout_data:
                    global_step_count += local_step_count
                    sharded_storages.append(sharded_storage)

                # Perform learning update
                with timer.time("learn_step_time"):
                    (learner_state, loss_info,) = learner_step_fn(
                        learner_state,
                        sharded_storages,
                    )
                learner_policy_version += 1

                # Broadcast updated parameters to actors
                with timer.time("params_queue_put_time"):
                    parameter_server.distribute_params(learner_state.params)

                # Periodic logging of training progress
                if learner_policy_version % learner_log_frequency == 0:
                    learner_timing_metrics = {
                        **timer.get_all_means(),
                        "update_no": learner_policy_version,
                        "timestep": global_step_count,
                        "learner_policy_version": learner_policy_version,
                        "learner_steps_per_seconds": int(
                            learner_policy_version / (time.perf_counter() - thread_start_time)
                        ),
                    }
                    logger.log(
                        learner_timing_metrics,
                        global_step_count,
                        learner_policy_version,
                        LogEvent.MISC,
                    )
                    logger.log(loss_info, global_step_count, learner_policy_version, LogEvent.TRAIN)

            # Submit policy for asynchronous evaluation if evaluation is enabled
            if num_evaluation > 0:
                rng_key, eval_key = split_key_fn(rng_key)
                async_evaluator.submit_evaluation(
                    learner_state, eval_key, eval_step, global_step_count
                )

    return learner_rollout


def get_learner_thread(
    config: DictConfig,
    learn_step: SebulbaLearnerFn[CoreLearnerState, ImpalaTransition],
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
            f"{Fore.MAGENTA}{Style.BRIGHT}Actor thread {actor_thread.name} has joined.{Style.RESET_ALL}"
        )


def learner_setup(
    env_factory: EnvFactory,
    keys: chex.Array,
    learner_devices: Sequence[jax.Device],
    config: DictConfig,
) -> Tuple[
    SebulbaLearnerFn[CoreLearnerState, ImpalaTransition],
    ActorCriticApply,
    CoreLearnerState,
]:
    """Setup learner networks and initial state."""

    # Get environment specifications
    env = env_factory(num_envs=1)
    num_actions = int(env.action_space().num_values)
    example_obs = env.observation_space().generate_value()
    config.system.action_dim = num_actions
    config.system.observation_shape = example_obs.shape
    env.close()

    key, actor_critic_net_key = keys

    # Build actor critic network
    shared_torso = hydra.utils.instantiate(config.network.shared_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    actor_critic_network = ActorCritic(
        input_layer=ArrayInput(),
        torso=shared_torso,
        action_head=actor_action_head,
        critic_head=critic_head,
    )

    # Configure learning rate schedules
    assert config.system.actor_lr == config.system.critic_lr, (
        "Actor and critic learning rates must be the same for IMPALA with a shared torso."
        "This is more just to ensure the user knows what learning rate is being used."
    )
    actor_critic_lr = make_learning_rate(
        config.system.actor_lr, config, 1, config.system.num_minibatches
    )

    # Setup optimizers with gradient clipping
    actor_critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm), optax.rmsprop(actor_critic_lr)
    )

    # Initialize network parameters
    init_x = example_obs
    init_x = jnp.expand_dims(init_x, axis=0)

    actor_critic_params = actor_critic_network.init(actor_critic_net_key, init_x)
    actor_critic_opt_state = actor_critic_optim.init(actor_critic_params)

    params = actor_critic_params

    # Extract network functions
    actor_critic_network_apply_fn = actor_critic_network.apply

    # Create and compile learner step function
    learn_step = get_learner_step_fn(
        actor_critic_network_apply_fn, actor_critic_optim.update, config
    )
    learn_step = jax.pmap(learn_step, axis_name="learner_devices", devices=learner_devices)

    # Load from checkpoint if specified
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # Initialize complete learner state
    learner_state = CoreLearnerState(params, actor_critic_opt_state, key)

    # Prepare learner state for multi-device training
    learner_state = flax.jax_utils.replicate(learner_state, devices=learner_devices)

    return learn_step, actor_critic_network_apply_fn, learner_state


def run_experiment(_config: DictConfig) -> float:
    """Run IMPALA experiment."""
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

    # Validate and adjust total timesteps and other config values
    config = check_total_timesteps(config)

    # Setup environment factory
    env_factory = environments.make_factory(config)
    assert isinstance(
        env_factory, EnvFactory
    ), "Environment factory must be an instance of EnvFactory"

    # Initialize random number generators
    key, key_e, actor_critic_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)
    np_rng = np.random.default_rng(config.arch.seed)

    # Setup learner and networks
    learn_step, actor_critic_apply_fn, learner_state = learner_setup(
        env_factory, (key, actor_critic_net_key), local_learner_devices, config
    )

    # Setup evaluation
    actor_apply_fn = lambda params, obs: actor_critic_apply_fn(params, obs)[0]
    eval_act_fn = get_distribution_act_fn(config, actor_apply_fn)
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
                actor_critic_apply_fn=actor_critic_apply_fn,
                rng_key=thread_key,
                config=config,
                seeds=seeds,
                logger=logger,
                learner_devices=local_learner_devices,
                thread_lifetime=actor_thread_lifetime,
            )
            print(
                f"{Fore.BLUE}{Style.BRIGHT}Starting actor thread {thread_name}...{Style.RESET_ALL}"
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

        logger.log(
            eval_metrics,
            int(config.arch.total_timesteps),
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
    config_name="default_ff_impala_shared_torso.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    start = time.perf_counter()
    eval_performance = run_experiment(cfg)
    end = time.perf_counter()
    print(
        f"{Fore.CYAN}{Style.BRIGHT}IMPALA experiment completed in "
        f"{end - start:.2f}s with a final episode return of {eval_performance}.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
