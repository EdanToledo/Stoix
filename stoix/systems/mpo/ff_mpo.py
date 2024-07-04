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
import rlax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ContinuousQApply,
    ExperimentOutput,
    LearnerFn,
    LogEnvState,
    OnlineAndTarget,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import CompositeNetwork
from stoix.networks.base import FeedForwardActor as Actor
from stoix.systems.mpo.discrete_loss import (
    categorical_mpo_loss,
    clip_categorical_mpo_params,
)
from stoix.systems.mpo.mpo_types import (
    CategoricalDualParams,
    MPOLearnerState,
    MPOOptStates,
    MPOParams,
    SequenceStep,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import (
    batch_n_step_bootstrapped_returns,
    batch_retrace_continuous,
    batch_truncated_generalized_advantage_estimation,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_warmup_fn(
    env: Environment,
    params: MPOParams,
    actor_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: LogEnvState, timesteps: TimeStep, buffer_states: BufferState, keys: chex.PRNGKey
    ) -> Tuple[LogEnvState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, chex.PRNGKey], SequenceStep]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params.online, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            sequence_step = SequenceStep(
                last_timestep.observation, action, timestep.reward, done, truncated, log_prob, info
            )

            return (env_state, timestep, key), sequence_step

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
        # Swap the batch and time axes.
        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, ContinuousQApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[MPOLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, q_apply_fn = apply_fns
    actor_update_fn, q_update_fn, dual_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(learner_state: MPOLearnerState, _: Any) -> Tuple[MPOLearnerState, Tuple]:
        def _env_step(
            learner_state: MPOLearnerState, _: Any
        ) -> Tuple[MPOLearnerState, SequenceStep]:
            """Step the environment."""
            params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params.online, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            sequence_step = SequenceStep(
                last_timestep.observation, action, timestep.reward, done, truncated, log_prob, info
            )

            learner_state = MPOLearnerState(
                params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, sequence_step

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        # Swap the batch and time axes.
        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _actor_loss_fn(
                online_actor_params: FrozenDict,
                dual_params: CategoricalDualParams,
                target_actor_params: FrozenDict,
                target_q_params: FrozenDict,
                sequence: SequenceStep,
            ) -> chex.Array:
                # Reshape the observations to [B*T, ...].
                reshaped_obs = jax.tree_util.tree_map(
                    lambda x: merge_leading_dims(x, 2), sequence.obs
                )
                batch_length = sequence.action.shape[0] * sequence.action.shape[1]  # B*T

                online_actor_policy = actor_apply_fn(online_actor_params, reshaped_obs)
                target_actor_policy = actor_apply_fn(target_actor_params, reshaped_obs)
                # In discrete MPO, we evaluate all actions instead of sampling.
                a_improvement = jnp.arange(config.system.action_dim).astype(jnp.float32)
                a_improvement = jnp.tile(
                    a_improvement[..., jnp.newaxis], [1, batch_length]
                )  # [D, B*T]
                a_improvement = jax.nn.one_hot(a_improvement, config.system.action_dim)
                target_q_values = jax.vmap(q_apply_fn, in_axes=(None, None, 0))(
                    target_q_params, reshaped_obs, a_improvement
                )  # [D, B*T]

                # Compute the policy and dual loss.
                loss, loss_info = categorical_mpo_loss(
                    dual_params=dual_params,
                    online_action_distribution=online_actor_policy,
                    target_action_distribution=target_actor_policy,
                    q_values=target_q_values,
                    epsilon=config.system.epsilon,
                    epsilon_policy=config.system.epsilon_policy,
                )

                return jnp.mean(loss), loss_info

            def _q_loss_fn(
                online_q_params: FrozenDict,
                target_q_params: FrozenDict,
                online_actor_params: FrozenDict,
                target_actor_params: FrozenDict,
                sequence: SequenceStep,
                rng_key: chex.PRNGKey,
            ) -> jnp.ndarray:

                online_actor_policy = actor_apply_fn(
                    online_actor_params, sequence.obs
                )  # [B, T, ...]
                target_actor_policy = actor_apply_fn(
                    target_actor_params, sequence.obs
                )  # [B, T, ...]
                a_t = jax.nn.one_hot(sequence.action, config.system.action_dim)  # [B, T, ...]
                online_q_t = q_apply_fn(online_q_params, sequence.obs, a_t)  # [B, T]

                # Cast and clip rewards.
                discount = 1.0 - sequence.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    sequence.reward, -config.system.max_abs_reward, config.system.max_abs_reward
                ).astype(jnp.float32)

                # Policy to use for policy evaluation and bootstrapping.
                if config.system.use_online_policy_to_bootstrap:
                    policy_to_evaluate = online_actor_policy
                else:
                    policy_to_evaluate = target_actor_policy

                # Action(s) to use for policy evaluation; shape [N, B, T].
                if config.system.stochastic_policy_eval:
                    a_evaluation = policy_to_evaluate.sample(
                        seed=rng_key, sample_shape=config.system.num_samples
                    )  # [N, B, T, ...]
                else:
                    a_evaluation = policy_to_evaluate.mode()[jnp.newaxis, ...]  # [N=1, B, T, ...]

                # Add a stopgrad in case we use the online policy for evaluation.
                a_evaluation = jax.lax.stop_gradient(a_evaluation)
                a_evaluation = jax.nn.one_hot(a_evaluation, config.system.action_dim)

                # Compute the Q-values for the next state-action pairs; [N, B, T].
                q_values = jax.vmap(q_apply_fn, in_axes=(None, None, 0))(
                    target_q_params, sequence.obs, a_evaluation
                )

                # When policy_eval_stochastic == True, this corresponds to expected SARSA.
                # Otherwise, the mean is a no-op.
                v_t = jnp.mean(q_values, axis=0)  # [B, T]

                if config.system.use_retrace:
                    # Compute the log-rhos for the retrace targets.
                    log_rhos = target_actor_policy.log_prob(sequence.action) - sequence.log_prob

                    # Compute target Q-values
                    target_q_t = q_apply_fn(target_q_params, sequence.obs, a_t)  # [B, T]

                    # Compute retrace targets.
                    # These targets use the rewards and discounts as in normal TD-learning but
                    # they use a mix of bootstrapped values V(s') and Q(s', a'), weighing the
                    # latter based on how likely a' is under the current policy (s' and a' are
                    # samples from replay).
                    # See [Munos et al., 2016](https://arxiv.org/abs/1606.02647) for more.
                    retrace_error = batch_retrace_continuous(
                        online_q_t[:, :-1],
                        target_q_t[:, 1:-1],
                        v_t[:, 1:],
                        r_t[:, :-1],
                        d_t[:, :-1],
                        log_rhos[:, 1:-1],
                        config.system.retrace_lambda,
                    )
                    q_loss = rlax.l2_loss(retrace_error).mean()
                elif config.system.use_n_step_bootstrap:
                    n_step_value_target = batch_n_step_bootstrapped_returns(
                        r_t[:, :-1],
                        d_t[:, :-1],
                        v_t[:, 1:],
                        config.system.n_step_for_sequence_bootstrap,
                    )
                    td_error = online_q_t[:, :-1] - n_step_value_target
                    q_loss = rlax.l2_loss(td_error).mean()
                else:
                    _, gae_value_target = batch_truncated_generalized_advantage_estimation(
                        r_t[:, :-1],
                        d_t[:, :-1],
                        config.system.gae_lambda,
                        v_t,
                        time_major=False,
                        truncation_flags=sequence.truncated[:, :-1],
                    )
                    td_error = online_q_t[:, :-1] - gae_value_target
                    q_loss = rlax.l2_loss(td_error).mean()

                loss_info = {
                    "q_loss": q_loss,
                }

                return q_loss, loss_info

            params, opt_states, buffer_state, key = update_state

            key, sample_key, q_key = jax.random.split(key, num=3)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: SequenceStep = sequence_sample.experience

            # CALCULATE ACTOR AND DUAL LOSS
            actor_dual_grad_fn = jax.grad(_actor_loss_fn, argnums=(0, 1), has_aux=True)
            actor_dual_grads, actor_dual_loss_info = actor_dual_grad_fn(
                params.actor_params.online,
                params.dual_params,
                params.actor_params.target,
                params.q_params.target,
                sequence,
            )

            # CALCULATE Q LOSS
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.q_params.online,
                params.q_params.target,
                params.actor_params.online,
                params.actor_params.target,
                sequence,
                q_key,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            actor_dual_grads, actor_dual_loss_info = jax.lax.pmean(
                (actor_dual_grads, actor_dual_loss_info), axis_name="batch"
            )
            # pmean over devices.
            actor_dual_grads, actor_dual_loss_info = jax.lax.pmean(
                (actor_dual_grads, actor_dual_loss_info), axis_name="device"
            )

            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            actor_grads, dual_grads = actor_dual_grads

            # UPDATE ACTOR PARAMS AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_online_params = optax.apply_updates(params.actor_params.online, actor_updates)

            # UPDATE DUAL PARAMS AND OPTIMISER STATE
            dual_updates, dual_new_opt_state = dual_update_fn(dual_grads, opt_states.dual_opt_state)
            dual_new_params = optax.apply_updates(params.dual_params, dual_updates)
            dual_new_params = clip_categorical_mpo_params(dual_new_params)

            # UPDATE Q PARAMS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states.q_opt_state)
            q_new_online_params = optax.apply_updates(params.q_params.online, q_updates)
            # Target network polyak update.
            (new_target_actor_params, new_target_q_params) = optax.incremental_update(
                (actor_new_online_params, q_new_online_params),
                (params.actor_params.target, params.q_params.target),
                config.system.tau,
            )

            actor_new_params = OnlineAndTarget(actor_new_online_params, new_target_actor_params)
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

            # PACK NEW PARAMS AND OPTIMISER STATE
            new_params = MPOParams(actor_new_params, q_new_params, dual_new_params)
            new_opt_state = MPOOptStates(actor_new_opt_state, q_new_opt_state, dual_new_opt_state)

            # PACK LOSS INFO
            loss_info = {
                **actor_dual_loss_info,
                **q_loss_info,
            }
            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (params, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key = update_state
        learner_state = MPOLearnerState(
            params, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: MPOLearnerState) -> ExperimentOutput[MPOLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[MPOLearnerState], Actor, MPOLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions or action dimension from the environment.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, actor_net_key, q_net_key = keys

    # Define actor_network, q_network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=action_dim
    )
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    q_network_input = hydra.utils.instantiate(config.network.q_network.input_layer)
    q_network_torso = hydra.utils.instantiate(config.network.q_network.pre_torso)
    q_network_head = hydra.utils.instantiate(config.network.q_network.critic_head)
    q_network = CompositeNetwork([q_network_input, q_network_torso, q_network_head])

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs)
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    init_a = env.action_spec().generate_value()
    init_a = jax.nn.one_hot(init_a, action_dim)
    init_a = jax.tree_util.tree_map(lambda x: x[None, ...], init_a)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    target_actor_params = actor_params
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise q params and optimiser state.
    online_q_params = q_network.init(q_net_key, init_x, init_a)
    target_q_params = online_q_params
    q_opt_state = q_optim.init(online_q_params)

    # Initialise MPO Dual params and optimiser state.
    log_temperature = jnp.full([1], config.system.init_log_temperature, dtype=jnp.float32)

    log_alpha = jnp.full([1], config.system.init_log_alpha, dtype=jnp.float32)

    dual_params = CategoricalDualParams(
        log_temperature=log_temperature,
        log_alpha=log_alpha,
    )

    dual_lr = make_learning_rate(config.system.dual_lr, config, config.system.epochs)
    dual_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(dual_lr, eps=1e-5),
    )
    dual_opt_state = dual_optim.init(dual_params)

    params = MPOParams(
        OnlineAndTarget(actor_params, target_actor_params),
        OnlineAndTarget(online_q_params, target_q_params),
        dual_params,
    )
    opt_states = MPOOptStates(actor_opt_state, q_opt_state, dual_opt_state)

    actor_network_apply_fn = actor_network.apply
    q_network_apply_fn = q_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, q_network_apply_fn)
    update_fns = (actor_optim.update, q_optim.update, dual_optim.update)

    # Create replay buffer
    dummy_sequence_step = SequenceStep(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        truncated=jnp.zeros((), dtype=bool),
        log_prob=jnp.zeros((), dtype=float),
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
    buffer_fn = fbx.make_trajectory_buffer(
        max_size=config.system.buffer_size,
        min_length_time_axis=config.system.sample_sequence_length,
        sample_batch_size=config.system.batch_size,
        sample_sequence_length=config.system.sample_sequence_length,
        period=config.system.period,
        add_batch_size=config.arch.num_envs,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_sequence_step)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, actor_network_apply_fn, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
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
        restored_params, _ = loaded_checkpoint.restore_params(TParams=MPOParams)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (params, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, buffer_states = replicate_learner
    # Warmup the buffer.
    env_states, timesteps, keys, buffer_states = warmup(
        env_states, timesteps, buffer_states, warmup_keys
    )
    init_learner_state = MPOLearnerState(
        params, opt_states, buffer_states, step_keys, env_states, timesteps
    )

    return learn, actor_network, init_learner_state


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
    key, key_e, actor_net_key, q_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, q_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, actor_network.apply),
        params=learner_state.params.actor_params.online,
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
    best_params = unreplicate_batch_dim(learner_state.params.actor_params.online)
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
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.actor_params.online
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


@hydra.main(config_path="../../configs", config_name="default_ff_mpo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}MPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
