import copy
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from rlax import SIGNED_HYPERBOLIC_PAIR, TxPair

from stoix.base_types import (
    ActorApply,
    AnakinExperimentOutput,
    LearnerFn,
    LogEnvState,
    OnlineAndTarget,
    RNNOffPolicyLearnerState,
)
from stoix.evaluator import evaluator_setup, get_rec_distribution_act_fn
from stoix.networks.base import RecurrentActor, ScannedRNN
from stoix.systems.q_learning.dqn_types import RNNTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_warmup_fn(
    env: Environment,
    q_params: OnlineAndTarget,
    q_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
    init_hstates: chex.Array,
) -> Callable:
    """Warmup function for R2D2.

    Steps the environment for config.system.warmup_steps using the recurrent Q-network.
    """

    def warmup(
        env_states: LogEnvState,
        timesteps: TimeStep,
        buffer_state: BufferState,
        keys: chex.PRNGKey,
        dones: chex.Array,
        truncateds: chex.Array,
        hstates: chex.Array,
    ) -> Tuple[
        LogEnvState, TimeStep, BufferState, chex.PRNGKey, chex.Array, chex.Array, chex.Array
    ]:
        def _env_step(
            carry: Tuple[chex.PRNGKey, LogEnvState, TimeStep, chex.Array, chex.Array, chex.Array],
            _: Any,
        ) -> Tuple[
            Tuple[chex.PRNGKey, LogEnvState, TimeStep, chex.Array, chex.Array, chex.Array],
            RNNTransition,
        ]:
            (
                key,
                env_state,
                last_timestep,
                last_done,
                last_truncated,
                last_hstates,
            ) = carry

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], last_timestep.observation
            )
            reset_hidden_state = jnp.logical_or(last_done, last_truncated)
            ac_in = (
                batched_observation,
                reset_hidden_state[jnp.newaxis, :],
            )

            # Run the network.
            new_hstate, actor_policy = actor_apply_fn(params.online, last_hstates, ac_in)

            # Sample action from the policy
            action = actor_policy.sample(seed=policy_key)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # log episode return and length
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            # Create transition
            transition = RNNTransition(
                last_timestep.observation,
                action,
                timestep.reward,
                done,
                truncated,
                timestep.extras["next_obs"],
                info,
                last_hstates,
            )

            # Update carry
            new_carry = (key, env_state, timestep, done, truncated, new_hstate)
            return new_carry, transition

        (
            new_keys,
            new_env_states,
            new_timesteps,
            new_done,
            new_truncated,
            new_hstates,
        ), traj_batch = jax.lax.scan(
            _env_step,
            (keys, env_states, timesteps, dones, truncateds, hstates),
            None,
            config.system.warmup_steps,
        )
        buffer_state = buffer_add_fn(buffer_state, traj_batch)
        return (
            new_env_states,
            new_timesteps,
            buffer_state,
            new_keys,
            new_done,
            new_truncated,
            new_hstates,
        )

    batched_warmup = jax.vmap(
        warmup,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0, 0, 0, 0, 0, 0),
        axis_name="batch",
    )
    return batched_warmup


def get_learner_fn(
    env: Environment,
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    buffer_fns: Tuple[Callable, Callable, Callable],
    transform_pair: TxPair,
    config: DictConfig,
) -> LearnerFn[RNNOffPolicyLearnerState]:
    """Get the learner function for R2D2.

    Implements prioritized double Q-learning with recurrent network.
    """
    buffer_add_fn, buffer_sample_fn, buffer_set_priorities = buffer_fns

    def _update_step(
        learner_state: RNNOffPolicyLearnerState, _: Any
    ) -> Tuple[RNNOffPolicyLearnerState, Tuple]:
        # Unpack state
        (
            q_params,
            opt_states,
            buffer_state,
            key,
            env_state,
            last_timestep,
            done,
            truncated,
            hstates,
        ) = learner_state

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, RNNPPOTransition]:
            """Step the environment."""
            (
                params,
                opt_states,
                buffer_state,
                key,
                env_state,
                last_timestep,
                last_done,
                last_truncated,
                last_hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], last_timestep.observation
            )
            reset_hidden_state = jnp.logical_or(last_done, last_truncated)
            ac_in = (
                batched_observation,
                reset_hidden_state[jnp.newaxis, :],
            )

            # Run the network.
            new_hidden_state, actor_policy = q_apply_fn(
                params.actor_params, hstates.policy_hidden_state, ac_in
            )

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # log episode return and length
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = RNNTransition(
                last_timestep.observation,
                action,
                timestep.reward,
                done,
                truncated,
                timestep.extras["next_obs"],
                info,
                last_hstates,
            )
            learner_state = RNNOffPolicyLearnerState(
                params,
                opt_states,
                buffer_state,
                key,
                env_state,
                timestep,
                done,
                truncated,
                new_hidden_state,
            )
            return learner_state, transition

        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Unpack new learner state
        (
            q_params,
            opt_states,
            buffer_state,
            key,
            env_state,
            timestep,
            done,
            truncated,
            hstates,
        ) = learner_state
        # Add the collected trajectory to the replay buffer
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any):
            params, opt_states, buffer_state, key = update_state
            key, sample_key = jax.random.split(key)
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transitions = transition_sample.experience

            def _q_loss_fn(
                q_params: FrozenDict,
                target_q_params: FrozenDict,
                transitions: RNNTransition,
                transition_probs: chex.Array,
                noise_key: chex.PRNGKey,
                importance_sampling_exponent: float,
            ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
                # Get sequence dimensions
                sequence_length = transitions.obs.shape[0]
                burn_in_length = config.system.burn_in_length
                learning_length = sequence_length - burn_in_length

                # Reset hidden states based on done flags
                reset_hidden_state = transitions.done
                obs_and_done = (transitions.obs, reset_hidden_state)
                next_obs_and_done = (transitions.next_obs, reset_hidden_state)

                # Initialize hidden states to zero for the sequence
                init_hstate = jnp.zeros_like(transitions.obs[0])

                # Run burn-in period (without computing loss)
                def burn_in_step(carry, inputs):
                    hstate, (obs, done) = carry, inputs
                    new_hstate, _ = q_apply_fn(q_params, hstate, (obs, done))
                    return (new_hstate, None)

                # Run burn-in to get initial hidden state
                burned_in_hstate, _ = jax.lax.scan(
                    burn_in_step,
                    init_hstate,
                    (transitions.obs[:burn_in_length], reset_hidden_state[:burn_in_length]),
                )

                # Get Q-values for learning period
                _, q_dist = q_apply_fn(
                    q_params,
                    burned_in_hstate,
                    (transitions.obs[burn_in_length:], reset_hidden_state[burn_in_length:]),
                )
                q_value = q_dist.preferences
                q_value = jnp.sum(
                    q_value
                    * jax.nn.one_hot(transitions.action[burn_in_length:], q_value.shape[-1]),
                    axis=-1,
                )

                # Get target Q-values for next observations
                _, target_q_dist = q_apply_fn(
                    target_q_params,
                    burned_in_hstate,
                    (transitions.next_obs[burn_in_length:], reset_hidden_state[burn_in_length:]),
                )
                target_q = jnp.max(target_q_dist.preferences, axis=-1)

                # Calculate n-step returns with transformed values
                n_step = config.system.n_step
                gamma = config.system.gamma
                r = transitions.reward[burn_in_length:]
                done = transitions.done[burn_in_length:].astype(jnp.float32)

                # Transform values using transform pair
                transformed_target_q = transform_pair.apply(target_q)
                transformed_q = transform_pair.apply(q_value)

                # Calculate TD target in transformed space
                transformed_returns = r + (gamma**n_step) * transformed_target_q * (1.0 - done)
                td_error = transformed_returns - transformed_q

                # Calculate sequence priorities using both max and mean TD errors
                sequence_errors = jnp.abs(td_error).astype(q_value.dtype)
                max_sequence_error = jnp.max(sequence_errors, axis=0)
                mean_sequence_error = jnp.mean(sequence_errors, axis=0)

                # Combine max and mean errors for priority calculation
                # η controls the balance between max and mean
                eta = config.system.priority_eta  # e.g., 0.9
                new_priorities = eta * max_sequence_error + (1 - eta) * mean_sequence_error

                # Apply importance sampling weights with epsilon for stability
                importance_weights = (
                    1.0 / (transition_probs[burn_in_length:] + 1e-6)  # Add epsilon
                ) ** importance_sampling_exponent
                importance_weights /= jnp.max(importance_weights)

                # Use Huber loss on transformed values
                weighted_loss = jnp.mean(
                    importance_weights
                    * jax.nn.huber_loss(td_error, delta=config.system.huber_loss_delta)
                )

                return weighted_loss, {
                    "q_loss": weighted_loss,
                    "priorities": new_priorities,
                    "mean_q": jnp.mean(transform_pair.apply_inv(transformed_q)),
                    "max_priority": jnp.max(new_priorities),
                    "mean_priority": jnp.mean(new_priorities),
                    "max_td_error": jnp.mean(max_sequence_error),
                    "mean_td_error": jnp.mean(mean_sequence_error),
                }

            noise_key = key
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, loss_info = q_grad_fn(
                params.online,
                params.target,
                transitions,
                transition_sample.priorities,
                noise_key,
                config.system.importance_sampling_exponent,
            )
            updated_priorities = loss_info.pop("priorities")
            buffer_state = buffer_set_priorities(
                buffer_state, transition_sample.indices, updated_priorities
            )
            q_grads, loss_info = jax.lax.pmean((q_grads, loss_info), axis_name="batch")
            q_grads, loss_info = jax.lax.pmean((q_grads, loss_info), axis_name="device")
            q_updates, new_opt_state = q_update_fn(q_grads, opt_states)
            new_online = optax.apply_updates(params.online, q_updates)
            new_target = optax.incremental_update(new_online, params.target, config.system.tau)
            new_params = OnlineAndTarget(online=new_online, target=new_target)
            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (q_params, opt_states, buffer_state, key)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )
        new_params, new_opt_states, buffer_state, key = update_state
        new_learner_state = (
            new_params,
            new_opt_states,
            buffer_state,
            key,
            env_state,
            timestep,
            done,
            truncated,
            hstates,
        )
        metric = traj_batch.info  # using the trajectory info for logging
        return new_learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: RNNOffPolicyLearnerState,
    ) -> AnakinExperimentOutput[RNNOffPolicyLearnerState]:
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")
        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNOffPolicyLearnerState], RecurrentActor, RNNOffPolicyLearnerState]:
    """Initialize learner function, network, optimizer, environment and states."""
    n_devices = len(jax.devices())

    # Get number of actions.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, q_net_key = keys

    # Instantiate the recurrent Q-network via Hydra
    q_network_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
    )
    q_network = RecurrentActor(
        pre_torso=q_network_pre_torso,
        hidden_state_dim=config.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.actor_network.rnn_layer.cell_type,
        post_torso=q_network_post_torso,
        action_head=q_network_action_head,
    )
    actor_rnn = ScannedRNN(
        hidden_state_dim=config.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.actor_network.rnn_layer.cell_type,
    )
    eval_q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.evaluation_epsilon,
    )
    eval_q_network = RecurrentActor(
        pre_torso=q_network_pre_torso,
        hidden_state_dim=config.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.actor_network.rnn_layer.cell_type,
        post_torso=q_network_post_torso,
        action_head=eval_q_network_action_head,
    )

    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise observation
    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
    init_x = (init_obs, init_done)

    # Initialize the recurrent hidden state
    init_hstates = actor_rnn.initialize_carry(batch_size=config.arch.num_envs)

    # Initialize Q-parameters
    q_online_params = q_network.init(q_net_key, init_hstates, init_x)
    q_target_params = q_online_params
    q_opt_state = q_optim.init(q_online_params)

    params = OnlineAndTarget(q_online_params, q_target_params)
    opt_states = q_opt_state

    q_network_apply_fn = q_network.apply

    # Create prioritized replay buffer
    dummy_transition = RNNTransition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0, 1), init_obs),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0, 1), init_obs),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )
    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )

    buffer_fn = fbx.make_prioritised_trajectory_buffer(
        max_size=config.system.buffer_size,
        min_length_time_axis=config.system.sample_sequence_length,
        sample_batch_size=config.system.batch_size,
        add_batch_size=config.arch.num_envs,
        sample_sequence_length=config.system.sample_sequence_length,
        period=1,
        priority_exponent=config.system.priority_exponent,
        device="gpu",
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample, buffer_fn.set_priorities)
    buffer_states = buffer_fn.init(dummy_transition)

    # Initialize transform pair for value rescaling
    q_tx_pair = SIGNED_HYPERBOLIC_PAIR

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, q_network_apply_fn, q_optim.update, buffer_fns, q_tx_pair, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, q_network_apply_fn, buffer_fn.add, config, init_hstates)
    warmup = jax.pmap(warmup, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )

    def reshape_states(x: chex.Array) -> chex.Array:
        return x.reshape(
            (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
        )

    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(TParams=OnlineAndTarget)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)

    def reshape_keys(x: chex.Array) -> chex.Array:
        return x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])

    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (params, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)

    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, buffer_states = replicate_learner

    # Warmup the replay buffer
    env_states, timesteps, buffer_states, step_keys, done, truncated, hstates = warmup(
        env_states, timesteps, buffer_states, warmup_keys, init_hstates
    )
    init_learner_state = RNNOffPolicyLearnerState(
        params,
        opt_states,
        buffer_states,
        step_keys,
        env_states,
        timesteps,
        done,
        truncated,
        hstates,
    )

    return learn, eval_q_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, q_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)

    # Setup learner.
    learn, eval_q_network, learner_state = learner_setup(env, (key, q_net_key), config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_rec_distribution_act_fn(config, eval_q_network.apply),
        params=learner_state.params.online,
        config=config,
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(-1e6)
    best_params = unreplicate_batch_dim(learner_state.params.online)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (config.system.epochs)
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.online
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_rec_r2d2.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}R2D2 experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
