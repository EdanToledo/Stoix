import copy
import time
from functools import partial
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from rlax import SIGNED_HYPERBOLIC_PAIR, TxPair
from stoa import Environment, TimeStep, WrapperState, get_final_step_metrics

from stoix.base_types import (
    ActorApply,
    AnakinExperimentOutput,
    LearnerFn,
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


def get_warmup_fn(
    env: Environment,
    q_params: OnlineAndTarget,
    q_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    """Get the warmup function for initializing the replay buffer."""

    def warmup(
        env_states: WrapperState,
        timesteps: TimeStep,
        keys: chex.PRNGKey,
        buffer_states: BufferState,
        hstates: chex.Array,
        dones: chex.Array,
        truncateds: chex.Array,
    ) -> Tuple[
        WrapperState, TimeStep, BufferState, chex.PRNGKey, chex.Array, chex.Array, chex.Array
    ]:
        def _env_step(
            carry: Tuple[chex.PRNGKey, WrapperState, TimeStep, chex.Array, chex.Array, chex.Array],
            _: Any,
        ) -> Tuple[
            Tuple[chex.PRNGKey, WrapperState, TimeStep, chex.Array, chex.Array, chex.Array],
            RNNTransition,
        ]:
            """Step the environment."""
            # UNPACK CARRY
            (
                key,
                env_state,
                last_timestep,
                last_done,
                last_truncated,
                last_hstates,
            ) = carry

            # SELECT ACTION
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

            # RUN NETWORK
            new_hstate, actor_policy = q_apply_fn(q_params.online, last_hstates, ac_in)

            # Sample action from the policy
            action = actor_policy.sample(seed=policy_key)
            action = action.squeeze(0)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # CREATE TRANSITION
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]
            transition = RNNTransition(
                last_timestep.observation,
                action,
                timestep.reward,
                reset_hidden_state,
                done,
                truncated,
                info,
                last_hstates,
            )

            # UPDATE CARRY
            new_carry = (key, env_state, timestep, done, truncated, new_hstate)
            return new_carry, transition

        # STEP ENVIRONMENT FOR WARMUP LENGTH
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

        # ADD TRAJECTORY TO BUFFER
        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return (
            new_env_states,
            new_timesteps,
            new_keys,
            buffer_states,
            new_hstates,
            new_done,
            new_truncated,
        )

    # VECTORIZE WARMUP FUNCTION
    batched_warmup: Callable = jax.vmap(
        warmup,
        in_axes=(0, 0, 0, 0, 0, 0, 0),
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
    importance_weight_scheduler_fn: Callable,
    config: DictConfig,
) -> LearnerFn[RNNOffPolicyLearnerState]:
    """Get the learner function for R2D2 training.

    This function creates the main training loop that:
    1. Collects trajectories using the current policy
    2. Updates the replay buffer with new experiences
    3. Samples sequences from the buffer
    4. Updates network parameters using n-step returns and prioritized replay

    Key R2D2 Components:
    - Recurrent state handling with burn-in for state warmup
    - Prioritized sequence replay with overlapping trajectories
    - N-step returns with transformed value functions
    - Target network updates with polyak averaging
    - Importance sampling for PER bias correction

    Args:
        env: The environment to interact with.
        q_apply_fn: The Q-network apply function.
        q_update_fn: The optimizer update function.
        buffer_fns: Tuple of (add, sample, set_priorities) buffer functions.
        transform_pair: Value transformation functions.
        importance_weight_scheduler_fn: Function to schedule importance sampling weights.
        config: The experiment configuration.

    Returns:
        The main learner function that performs training updates.
    """
    buffer_add_fn, buffer_sample_fn, buffer_set_priorities = buffer_fns

    def _update_step(
        learner_state: RNNOffPolicyLearnerState, _: Any
    ) -> Tuple[RNNOffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: RNNOffPolicyLearnerState, _: Any
        ) -> Tuple[RNNOffPolicyLearnerState, RNNTransition]:
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
            new_hidden_state, actor_policy = q_apply_fn(params.online, last_hstates, ac_in)

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            action = action.squeeze(0)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.last() & (timestep.discount == 0.0)).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = RNNTransition(
                last_timestep.observation,
                action,
                timestep.reward,
                reset_hidden_state,
                done,
                truncated,
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

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Unpack new learner state
        (
            params,
            opt_states,
            buffer_state,
            key,
            env_state,
            timestep,
            done,
            truncated,
            hstates,
        ) = learner_state

        # Add the trajectory to the buffer.
        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _q_loss_fn(
                q_params: FrozenDict,
                target_q_params: FrozenDict,
                sequences: RNNTransition,
                sequences_probs: chex.Array,
                importance_sampling_exponent: float,
            ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

                # Split sequence into burn-in and learning segments
                # Burn-in is used to warm up the RNN hidden state without generating gradients
                burn_in_length = config.system.burn_in_length
                burn_data = jax.tree.map(lambda x: x[:burn_in_length], sequences)
                learn_data = jax.tree.map(lambda x: x[burn_in_length:], sequences)

                # Initialize RNN state from the start of the sequence
                # This ensures consistent state initialization across training steps
                init_hstate = sequences.hstate[0]

                # Run burn-in phase to get warmed up hidden states
                # stop_gradient ensures no gradients flow through the burn-in computation
                burn_ac_in = (burn_data.obs, burn_data.reset_hidden_state)
                online_burned_in_hstate, _ = jax.lax.stop_gradient(
                    q_apply_fn(q_params, init_hstate, burn_ac_in)
                )
                target_burned_in_hstate, _ = jax.lax.stop_gradient(
                    q_apply_fn(target_q_params, init_hstate, burn_ac_in)
                )

                # Get Q-values for learning period using burned-in hidden states
                learn_ac_in = (learn_data.obs, learn_data.reset_hidden_state)
                _, online_q_dist = q_apply_fn(
                    q_params,
                    online_burned_in_hstate,
                    learn_ac_in,
                )
                online_q_values = online_q_dist.preferences

                # Get target Q-values for next observations
                _, target_q_dist = q_apply_fn(target_q_params, target_burned_in_hstate, learn_ac_in)
                target_q_values = target_q_dist.preferences

                # Get value-selector actions from online Q-values for double Q-learning
                # This helps reduce overestimation bias in Q-learning
                selector_actions = jnp.argmax(online_q_values, axis=-1)

                # Cast and clip rewards for numerical stability
                discount = 1.0 - learn_data.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    learn_data.reward, -config.system.max_abs_reward, config.system.max_abs_reward
                ).astype(jnp.float32)

                # Compute n-step TD error with transformed values
                # This helps stabilize learning with value transformation
                batch_td_error_fn = jax.vmap(
                    partial(
                        rlax.transformed_n_step_q_learning,
                        n=config.system.n_step,
                        tx_pair=transform_pair,
                    ),
                    in_axes=1,
                    out_axes=1,
                )
                batch_td_error = batch_td_error_fn(
                    online_q_values[:-1],
                    learn_data.action[:-1],
                    target_q_values[1:],
                    selector_actions[1:],
                    r_t[:-1],
                    d_t[:-1],
                )
                batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)

                # Apply importance sampling for PER bias correction
                importance_weights = 1.0 / (sequences_probs + 1e-6)
                importance_weights **= importance_sampling_exponent
                importance_weights /= jnp.max(importance_weights)
                mean_loss = jnp.mean(importance_weights * batch_loss)

                # Calculate priorities as mixture of max and mean sequence errors
                # This balances between focusing on high-error transitions and maintaining diversity
                abs_td_error = jnp.abs(batch_td_error)
                max_priority = config.system.priority_eta * jnp.max(abs_td_error, axis=0)
                mean_priority = (1 - config.system.priority_eta) * jnp.mean(abs_td_error, axis=0)
                new_priorities = max_priority + mean_priority

                return mean_loss, {
                    "q_loss": mean_loss,
                    "priorities": new_priorities,
                    "mean_q": jnp.mean(online_q_values),
                    "max_priority": jnp.max(new_priorities),
                    "mean_priority": jnp.mean(new_priorities),
                    "max_seq_td_error": jnp.mean(jnp.max(abs_td_error, axis=0)),
                    "mean_seq_td_error": jnp.mean(jnp.mean(abs_td_error, axis=0)),
                }

            params, opt_states, buffer_state, key = update_state

            key, sample_key = jax.random.split(key)

            # SAMPLE SEQUENCES
            sequences_sample = buffer_sample_fn(buffer_state, sample_key)
            sequences: RNNTransition = sequences_sample.experience  # [B, T, ...]
            sequences_probabilities = sequences_sample.probabilities
            sequences_indices = sequences_sample.indices

            # Convert B x T -> T x B
            sequences = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), sequences)

            # Get update step count
            step_count = optax.tree_utils.tree_get(opt_states, "count")
            importance_sampling_exponent = importance_weight_scheduler_fn(step_count)

            # Get gradients and loss
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.online,
                params.target,
                sequences,
                sequences_probabilities,
                importance_sampling_exponent,
            )

            # Update priorities in the buffer.
            updated_priorities = q_loss_info.pop("priorities")
            buffer_state = buffer_set_priorities(
                buffer_state, sequences_indices, updated_priorities
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            # UPDATE Q PARAMS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states)
            q_new_online_params = optax.apply_updates(params.online, q_updates)
            # Target network polyak update.
            new_target_q_params = optax.incremental_update(
                q_new_online_params, params.target, config.system.tau
            )
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

            # PACK LOSS INFO
            loss_info = {
                **q_loss_info,
            }
            return (q_new_params, q_new_opt_state, buffer_state, key), loss_info

        # Create update state
        update_state = (params, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )
        # Extract updated state
        params, opt_states, buffer_state, key = update_state
        # Create new learner state
        learner_state = RNNOffPolicyLearnerState(
            params, opt_states, buffer_state, key, env_state, timestep, done, truncated, hstates
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: RNNOffPolicyLearnerState,
    ) -> AnakinExperimentOutput[RNNOffPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        """

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
) -> Tuple[
    LearnerFn[RNNOffPolicyLearnerState], RecurrentActor, RNNOffPolicyLearnerState, ScannedRNN
]:
    """Initialize networks, optimizers and states for R2D2 training.

    This function sets up all necessary components for R2D2 training:
    1. Network Architecture:
       - Pre-torso for observation processing
       - RNN layer for temporal dependencies
       - Post-torso for feature processing
       - Action head for Q-value estimation
    2. Optimizer Configuration:
       - Gradient clipping for stability
       - Adam optimizer with configurable learning rate
    3. Buffer Setup:
       - Prioritized sequence replay buffer
       - Configurable sequence length and overlap
       - Burn-in period handling
    4. Evaluation Setup:
       - Separate evaluation network with different epsilon
       - Scanned RNN for efficient inference

    Args:
        env: The environment to interact with.
        keys: PRNG keys for initialization.
        config: The experiment configuration.

    Returns:
        A tuple containing:
        - learn_fn: The main learning function
        - eval_q_network: Network for evaluation
        - init_learner_state: Initial state for training
        - actor_rnn: Scanned RNN for efficient inference
    """
    # GET DEVICE INFO
    n_devices = len(jax.devices())

    # GET ACTION SPACE INFO
    action_dim = int(env.action_space().num_values)
    config.system.action_dim = action_dim

    # INITIALIZE PRNG KEYS
    key, q_net_key = keys

    # DEFINE NETWORKS
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

    # DEFINE EVALUATION NETWORK
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

    # DEFINE OPTIMIZERS
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # SETUP IMPORTANCE SAMPLING SCHEDULER
    importance_sampling_exponent_scheduler: Callable = optax.linear_schedule(
        init_value=config.system.importance_sampling_exponent,
        end_value=1.0,
        transition_steps=config.arch.num_updates * config.system.epochs,
        transition_begin=0,
    )

    # INITIALIZE OBSERVATIONS
    init_obs = env.observation_space().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )  # Give it num envs batch dimension
    init_obs = jax.tree_util.tree_map(
        lambda x: x[jnp.newaxis, ...], init_obs
    )  # Give it time dimension
    init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
    init_x = (init_obs, init_done)

    # INITIALIZE HIDDEN STATES
    init_hstates = actor_rnn.initialize_carry(batch_size=config.arch.num_envs)

    # INITIALIZE NETWORK PARAMETERS AND OPTIMIZER STATE
    q_online_params = q_network.init(q_net_key, init_hstates, init_x)
    q_target_params = q_online_params
    q_opt_state = q_optim.init(q_online_params)

    params = OnlineAndTarget(q_online_params, q_target_params)
    opt_states = q_opt_state

    # PACK NETWORK FUNCTIONS AND SCHEDULER
    q_network_apply_fn = q_network.apply
    apply_fns = q_network_apply_fn
    update_fns = q_optim.update
    scheduler_fns = importance_sampling_exponent_scheduler

    # SETUP REPLAY BUFFER
    dummy_transition = RNNTransition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0)[0], init_obs),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        reset_hidden_state=jnp.zeros((), dtype=bool),
        done=jnp.zeros((), dtype=bool),
        truncated=jnp.zeros((), dtype=bool),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
        hstate=jnp.zeros_like(init_hstates)[0],
    )

    # VALIDATE BUFFER CONFIG
    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    # Validate R2D2 specific parameters
    assert config.system.burn_in_length < config.system.sample_sequence_length, (
        f"{Fore.RED}{Style.BRIGHT}The burn-in length must be less than "
        + "the sample sequence length!{Style.RESET_ALL}"
    )
    assert config.system.period <= config.system.rollout_length, (
        f"{Fore.RED}{Style.BRIGHT}The period must be less than or equal to "
        + "the rollout length!{Style.RESET_ALL}"
    )
    # COMPUTE BUFFER AND BATCH SIZES PER DEVICE AND VECTORIZED UPDATE
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )

    # CREATE BUFFER
    buffer_fn = fbx.make_prioritised_trajectory_buffer(
        max_size=config.system.buffer_size,
        min_length_time_axis=config.system.sample_sequence_length,
        sample_batch_size=config.system.batch_size,
        add_batch_size=config.arch.num_envs,
        sample_sequence_length=config.system.sample_sequence_length,
        period=config.system.period,
        priority_exponent=config.system.priority_exponent,
        device="gpu",
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample, buffer_fn.set_priorities)
    buffer_states = buffer_fn.init(dummy_transition)

    # SETUP VALUE TRANSFORM
    q_tx_pair = SIGNED_HYPERBOLIC_PAIR

    # SETUP LEARNER AND WARMUP
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, q_tx_pair, scheduler_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, apply_fns, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    # INITIALIZE ENVIRONMENT STATES
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))

    def reshape_states(x: chex.Array) -> chex.Array:
        return x.reshape(
            (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
        )

    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # LOAD CHECKPOINT IF SPECIFIED
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # SETUP REPLICATED STATES
    dones = jnp.zeros((config.arch.num_envs,), dtype=bool)
    truncated = jnp.zeros((config.arch.num_envs,), dtype=bool)
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)

    def reshape_keys(x: chex.PRNGKey) -> chex.PRNGKey:
        return x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])

    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    # REPLICATE LEARNER STATE
    replicate_learner = (params, opt_states, buffer_states, init_hstates, dones, truncated)

    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)

    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # INITIALIZE FINAL LEARNER STATE
    params, opt_states, buffer_states, hstates, dones, truncated = replicate_learner
    env_states, timesteps, keys, buffer_states, hstates, dones, truncated = warmup(
        env_states, timesteps, warmup_keys, buffer_states, hstates, dones, truncated
    )
    init_learner_state = RNNOffPolicyLearnerState(
        params,
        opt_states,
        buffer_states,
        step_keys,
        env_states,
        timesteps,
        dones,
        truncated,
        hstates,
    )

    return learn, eval_q_network, init_learner_state, actor_rnn


def run_experiment(_config: DictConfig) -> float:
    """Run the R2D2 training experiment."""
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
    learn, eval_q_network, learner_state, actor_rnn = learner_setup(env, (key, q_net_key), config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_rec_distribution_act_fn(config, eval_q_network.apply),
        params=learner_state.params.online,
        config=config,
        use_recurrent_net=True,
        scanned_rnn=actor_rnn,
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
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
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
