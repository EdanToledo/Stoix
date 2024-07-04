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
    Observation,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import CompositeNetwork
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import MultiNetwork
from stoix.networks.postprocessors import tanh_to_spec
from stoix.systems.ddpg.ddpg_types import DDPGOptStates, DDPGParams
from stoix.systems.q_learning.dqn_types import Transition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_default_behavior_policy(config: DictConfig, actor_apply_fn: ActorApply) -> Callable:
    def behavior_policy(
        params: DDPGParams, observation: Observation, key: chex.PRNGKey
    ) -> chex.Array:
        action = actor_apply_fn(params, observation).mode()
        action_scale = (config.system.action_maximum - config.system.action_minimum) / 2
        if config.system.exploration_noise != 0:
            action = rlax.add_gaussian_noise(
                key, action, config.system.exploration_noise * action_scale
            ).clip(config.system.action_minimum, config.system.action_maximum)
        return action

    return behavior_policy


def get_warmup_fn(
    env: Environment,
    params: DDPGParams,
    actor_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:

    exploratory_actor_apply = get_default_behavior_policy(config, actor_apply_fn)

    def warmup(
        env_states: LogEnvState, timesteps: TimeStep, buffer_states: BufferState, keys: chex.PRNGKey
    ) -> Tuple[LogEnvState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, chex.PRNGKey], Transition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            action = exploratory_actor_apply(
                params.actor_params.online, last_timestep.observation, policy_key
            )

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            return (env_state, timestep, key), transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, ContinuousQApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, q_apply_fn = apply_fns
    actor_update_fn, q_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns
    exploratory_actor_apply = get_default_behavior_policy(config, actor_apply_fn)

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            action = exploratory_actor_apply(
                params.actor_params.online, last_timestep.observation, policy_key
            )

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            learner_state = OffPolicyLearnerState(
                params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _q_loss_fn(
                q_params: FrozenDict,
                target_q_params: FrozenDict,
                target_actor_params: FrozenDict,
                transitions: Transition,
                rng_key: chex.PRNGKey,
            ) -> jnp.ndarray:

                q_tm1 = q_apply_fn(q_params, transitions.obs, transitions.action)
                action_scale = (config.system.action_maximum - config.system.action_minimum) / 2
                noise = (
                    jax.random.normal(rng_key, transitions.action.shape)
                    * config.system.policy_noise
                )
                clipped_noise = (
                    jnp.clip(noise, -config.system.noise_clip, config.system.noise_clip)
                    * action_scale
                )
                next_action = (
                    actor_apply_fn(target_actor_params, transitions.next_obs).mode() + clipped_noise
                )
                next_action = jnp.clip(
                    next_action, config.system.action_minimum, config.system.action_maximum
                )
                q_t = q_apply_fn(target_q_params, transitions.next_obs, next_action)
                next_v = jnp.min(q_t, axis=-1)

                # Cast and clip rewards.
                discount = 1.0 - transitions.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    transitions.reward, -config.system.max_abs_reward, config.system.max_abs_reward
                ).astype(jnp.float32)

                target_q = jax.lax.stop_gradient(r_t + d_t * next_v)
                q_error = q_tm1 - jnp.expand_dims(target_q, -1)
                q_loss = 0.5 * jnp.mean(jnp.square(q_error))

                loss_info = {
                    "q_loss": q_loss,
                    "q1_pred": jnp.mean(q_t[..., 0]),
                    "q2_pred": jnp.mean(q_t[..., 1]),
                }

                return q_loss, loss_info

            def _actor_loss_fn(
                actor_params: FrozenDict,
                q_params: FrozenDict,
                transitions: Transition,
            ) -> chex.Array:
                o_t = transitions.obs
                a_t = (
                    actor_apply_fn(actor_params, o_t)
                    .mode()
                    .clip(config.system.action_minimum, config.system.action_maximum)
                )
                q_value = q_apply_fn(q_params, o_t, a_t)

                actor_loss = -jnp.mean(q_value)

                loss_info = {
                    "actor_loss": actor_loss,
                }
                return actor_loss, loss_info

            params, opt_states, buffer_state, key = update_state

            key, sample_key = jax.random.split(key, num=2)

            # SAMPLE TRANSITIONS
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transitions: Transition = transition_sample.experience

            # CALCULATE ACTOR LOSS
            actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(
                params.actor_params.online,
                params.q_params.online,
                transitions,
            )

            # CALCULATE Q LOSS
            key, q_loss_key = jax.random.split(key)
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.q_params.online,
                params.q_params.target,
                params.actor_params.target,
                transitions,
                q_loss_key,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="batch"
            )
            # pmean over devices.
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="device"
            )

            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            # UPDATE ACTOR PARAMS AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_online_params = optax.apply_updates(params.actor_params.online, actor_updates)

            # UPDATE Q PARAMS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states.q_opt_state)
            q_new_online_params = optax.apply_updates(params.q_params.online, q_updates)
            # Target network polyak update.
            new_target_actor_params, new_target_q_params = optax.incremental_update(
                (actor_new_online_params, q_new_online_params),
                (params.actor_params.target, params.q_params.target),
                config.system.tau,
            )

            actor_new_params = OnlineAndTarget(actor_new_online_params, new_target_actor_params)
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

            # PACK NEW PARAMS AND OPTIMISER STATE
            new_params = DDPGParams(actor_new_params, q_new_params)
            new_opt_state = DDPGOptStates(actor_new_opt_state, q_new_opt_state)

            # PACK LOSS INFO
            loss_info = {
                **actor_loss_info,
                **q_loss_info,
            }
            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (params, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(
            params, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: OffPolicyLearnerState) -> ExperimentOutput[OffPolicyLearnerState]:
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
) -> Tuple[LearnerFn[OffPolicyLearnerState], Actor, OffPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions.
    action_dim = int(env.action_spec().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_spec().minimum)
    config.system.action_maximum = float(env.action_spec().maximum)

    # PRNG keys.
    key, actor_net_key, q_net_key = keys

    # Define actor_network, q_network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=action_dim
    )
    action_head_post_processor = hydra.utils.instantiate(
        config.network.actor_network.post_processor,
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
        scale_fn=tanh_to_spec,
    )
    actor_action_head = CompositeNetwork([actor_action_head, action_head_post_processor])
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    def create_q_network(cfg: DictConfig) -> CompositeNetwork:
        q_network_input = hydra.utils.instantiate(cfg.network.q_network.input_layer)
        q_network_torso = hydra.utils.instantiate(cfg.network.q_network.pre_torso)
        q_network_head = hydra.utils.instantiate(cfg.network.q_network.critic_head)
        return CompositeNetwork([q_network_input, q_network_torso, q_network_head])

    double_q_network = MultiNetwork([create_q_network(config), create_q_network(config)])

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs)
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)

    def delayed_policy_update(step_count: int) -> bool:
        should_update: bool = jnp.mod(step_count, config.system.policy_frequency) == 0
        return should_update

    actor_optim = optax.conditionally_mask(
        optax.chain(
            optax.clip_by_global_norm(config.system.max_grad_norm),
            optax.adam(actor_lr, eps=1e-5),
        ),
        should_transform_fn=delayed_policy_update,
    )
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    init_a = jnp.zeros((1, action_dim))

    # Initialise actor params and optimiser state.
    actor_online_params = actor_network.init(actor_net_key, init_x)
    actor_target_params = actor_online_params
    actor_opt_state = actor_optim.init(actor_online_params)

    actor_params = OnlineAndTarget(actor_online_params, actor_target_params)

    # Initialise critic params and optimiser state.
    q_online_params = double_q_network.init(q_net_key, init_x, init_a)
    q_target_params = q_online_params

    q_params = OnlineAndTarget(q_online_params, q_target_params)

    q_opt_state = q_optim.init(q_online_params)

    params = DDPGParams(actor_params, q_params)
    opt_states = DDPGOptStates(actor_opt_state, q_opt_state)

    actor_network_apply_fn = actor_network.apply
    q_network_apply_fn = double_q_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, q_network_apply_fn)
    update_fns = (actor_optim.update, q_optim.update)

    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((action_dim), dtype=float),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
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
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_transition)

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
        restored_params, _ = loaded_checkpoint.restore_params(TParams=DDPGParams)
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
    init_learner_state = OffPolicyLearnerState(
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
    max_episode_return = jnp.float32(-1e7)
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


@hydra.main(config_path="../../configs", config_name="default_ff_td3.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}TD3 experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
