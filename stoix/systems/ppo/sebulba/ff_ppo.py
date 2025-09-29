# flake8: noqa: CCR001
import copy
import os
import queue
import random
import threading
import time
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

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
from stoa.core_wrappers.episode_metrics import get_final_step_metrics

from stoix.base_types import (
    Action,
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CoreLearnerState,
    CriticApply,
    Done,
    LogProb,
    Observation,
    SebulbaExperimentOutput,
    SebulbaLearnerFn,
    Truncated,
    Value,
)
from stoix.evaluator import get_distribution_act_fn, get_sebulba_eval_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.networks.inputs import ArrayInput
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvFactory
from stoix.utils.jax_utils import merge_leading_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
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


class PPOTransition(NamedTuple):
    """Transition tuple for Sebulba PPO. We just remove the info field."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array


class AsyncEvaluator(AsyncEvaluatorBase):
    """PPO-specific asynchronous evaluator."""

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
                actor_params = jax.block_until_ready(unreplicated_learner_state.params.actor_params)

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
    apply_fns: Tuple[ActorApply, CriticApply]
) -> Callable[
    [ActorCriticParams, Observation, chex.PRNGKey],
    Tuple[Action, Value, LogProb, chex.PRNGKey],
]:
    """Create action function for actor threads."""
    actor_apply_fn, critic_apply_fn = apply_fns

    def actor_fn(
        params: ActorCriticParams, observation: Observation, rng_key: chex.PRNGKey
    ) -> Tuple[Action, Value, LogProb, chex.PRNGKey]:
        rng_key, policy_key = jax.random.split(rng_key)
        pi = actor_apply_fn(params.actor_params, observation)
        value = critic_apply_fn(params.critic_params, observation)
        action = pi.sample(seed=policy_key)
        log_prob = pi.log_prob(action)
        return action, value, log_prob, rng_key

    return actor_fn


def get_rollout_fn(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    parameter_server: "ParameterServer",
    rollout_pipeline: "OnPolicyPipeline",
    apply_fns: Tuple[ActorApply, CriticApply],
    config: DictConfig,
    logger: StoixLogger,
    learner_devices: Sequence[jax.Device],
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
) -> Callable[[chex.PRNGKey], None]:
    """Create rollout function for actor threads."""

    # Setup action function and data preparation
    act_fn = get_act_fn(apply_fns)
    act_fn = jax.jit(act_fn, device=actor_device)

    @jax.jit
    def prepare_data(storage: List[PPOTransition]) -> List[PPOTransition]:
        """Prepare and shard trajectory data for learner devices."""
        split_data: List[PPOTransition] = jax.tree.map(
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
        traj_storage: List[PPOTransition] = []
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
                            a_tm1, value_tm1, log_prob_tm1, rng_key = act_fn(
                                params, obs_tm1, rng_key
                            )

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
                                PPOTransition(
                                    obs=obs_tm1,
                                    done=done_t,
                                    truncated=trunc_t,
                                    action=a_tm1,
                                    value=value_tm1,
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
                    sharded_traj_storage = PPOTransition(
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
                        f"{Fore.MAGENTA}{Style.BRIGHT}Actor {thread_lifetime.id} has completed"
                        f"{num_rollouts} rollouts. Stopping...{Style.RESET_ALL}"
                    )
                    break

            envs.close()

    return rollout_fn


def get_actor_thread(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    parameter_server: ParameterServer,
    rollout_pipeline: OnPolicyPipeline,
    apply_fns: Tuple[ActorApply, CriticApply],
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
        apply_fns=apply_fns,
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
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> SebulbaLearnerFn[CoreLearnerState, PPOTransition]:
    """Create learner update function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: CoreLearnerState, sharded_traj_batchs: List[PPOTransition]
    ) -> Tuple[CoreLearnerState, Dict[str, Array]]:

        # Combine data from all actors
        traj_batch = jax.tree.map(lambda *x: jnp.hstack(x), *sharded_traj_batchs)

        # CALCULATE ADVANTAGE
        params, opt_states, key = learner_state

        r_t = traj_batch.reward[:-1]  # Rewards at t
        d_t = 1.0 - traj_batch.done[:-1].astype(jnp.float32)  # Discount at t
        d_t = d_t * config.system.gamma
        values = traj_batch.value  # all values

        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            values=values,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
        )

        chex.assert_shape(advantages, r_t.shape)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                o_tm1, a_tm1, v_tm1, log_prob_tm1, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    o_tm1: Array,
                    a_tm1: Array,
                    log_prob_tm1: LogProb,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    actor_policy = actor_apply_fn(actor_params, o_tm1)
                    log_prob = actor_policy.log_prob(a_tm1)

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(log_prob, log_prob_tm1, gae, config.system.clip_eps)
                    entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    o_tm1: Array,
                    v_tm1: Array,
                    targets: Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    pred_v_tm1 = critic_apply_fn(critic_params, o_tm1)

                    # CALCULATE VALUE LOSS
                    value_loss = clipped_value_loss(
                        pred_v_tm1, v_tm1, targets, config.system.clip_eps
                    )

                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return critic_total_loss, loss_info

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, o_tm1, a_tm1, log_prob_tm1, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params, o_tm1, v_tm1, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                # pmean over devices.
                actor_grads, critic_grads, actor_loss_info, critic_loss_info = jax.lax.pmean(
                    (actor_grads, critic_grads, actor_loss_info, critic_loss_info),
                    axis_name="learner_devices",
                )

                # Update actor parameters and optimizer state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # Update critic parameters and optimizer state
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = ActorCriticParams(actor_new_params, critic_new_params)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                }
                return (new_params, new_opt_state), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            # Extract data
            o_tm1 = traj_batch.obs[:-1]  # Observations at t - 1
            a_tm1 = traj_batch.action[:-1]  # Actions at t - 1
            v_tm1 = traj_batch.value[:-1]  # Values at t - 1
            log_prob_tm1 = traj_batch.log_prob[:-1]  # Log probabilities at t - 1

            # PREPARE AND SHUFFLE MINIBATCHES
            envs_per_batch = config.arch.total_num_envs // len(config.arch.learner.device_ids)
            batch = (o_tm1, a_tm1, v_tm1, log_prob_tm1, advantages, targets)
            chex.assert_tree_shape_prefix(batch, (config.system.rollout_length, envs_per_batch))
            batch = jax.tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree.map(lambda x: jax.random.permutation(shuffle_key, x), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # Update minibatches
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = CoreLearnerState(params, opt_states, key)

        return learner_state, loss_info

    def learner_step_fn(
        learner_state: CoreLearnerState, traj_batch: List[PPOTransition]
    ) -> SebulbaExperimentOutput[CoreLearnerState]:
        """Single learner step"""

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
    learn_step: SebulbaLearnerFn[CoreLearnerState, PPOTransition],
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
    SebulbaLearnerFn[CoreLearnerState, PPOTransition],
    Tuple[ActorApply, CriticApply],
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

    key, actor_net_key, critic_net_key = keys

    # Build actor and critic networks
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = Actor(
        input_layer=ArrayInput(), torso=actor_torso, action_head=actor_action_head
    )
    # dummy_critic_network = VisualResNetTorso(
    #     (1,), (1,), hidden_sizes=(1,)
    # )
    critic_network = Critic(input_layer=ArrayInput(), torso=critic_torso, critic_head=critic_head)

    # Configure learning rate schedules
    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    # Setup optimizers with gradient clipping
    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )

    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialize network parameters
    init_x = example_obs
    init_x = jnp.expand_dims(init_x, axis=0)

    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    params = ActorCriticParams(actor_params, critic_params)

    # Extract network functions
    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Create and compile learner step function
    learn_step = get_learner_step_fn(apply_fns, update_fns, config)
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
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    learner_state = CoreLearnerState(params, opt_states, key)

    # Prepare learner state for multi-device training
    learner_state = flax.jax_utils.replicate(learner_state, devices=learner_devices)

    return learn_step, apply_fns, learner_state


def run_experiment(_config: DictConfig) -> float:
    """Run PPO experiment."""
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
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )
    np_rng = np.random.default_rng(config.arch.seed)

    # Setup learner and networks
    learn_step, apply_fns, learner_state = learner_setup(
        env_factory, (key, actor_net_key, critic_net_key), local_learner_devices, config
    )

    # Setup evaluation
    actor_apply_fn, _ = apply_fns
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
                apply_fns=apply_fns,
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
    config_name="default_ff_ppo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    start = time.perf_counter()
    eval_performance = run_experiment(cfg)
    end = time.perf_counter()
    print(
        f"{Fore.CYAN}{Style.BRIGHT}PPO experiment completed in "
        f"{end - start:.2f}s with a final episode return of {eval_performance}.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
