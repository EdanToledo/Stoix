import copy
import functools
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import mctx
import optax
import rlax
import tensorflow_probability.substrates.jax as tfp
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CriticApply,
    ExperimentOutput,
    LearnerFn,
    LogEnvState,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.search.evaluator import search_evaluator_setup
from stoix.systems.search.search_types import (
    EnvironmentStep,
    RootFnApply,
    SampledExItTransition,
    SearchApply,
    ZLearnerState,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics

tfd = tfp.distributions

# This is a implementation of sampled muzero using a perfect environment model
# for search such as AlphaZero.


def add_gaussian_noise(
    key: chex.PRNGKey,
    action: chex.Array,
    exploration_fraction: float,
    action_min: float,
    action_max: float,
) -> chex.Array:
    """Returns continuous action with noise drawn from a Gaussian distribution.

    Args:
        key: a key from `jax.random`.
        action: continuous action scalar or vector.
        exploration_fraction: scale of the noise.
        action_min: minimum value of the action.
        action_max: maximum value of the action.

    Returns:
        noisy action, of the same shape as input action.
    """
    shape = action.shape
    noise = jax.random.truncated_normal(key, action_min, action_max, shape=shape)
    action = (1 - exploration_fraction) * action + exploration_fraction * noise
    return jnp.clip(action, action_min, action_max)


def make_root_fn(
    actor_apply_fn: ActorApply,
    critic_apply_fn: CriticApply,
    config: DictConfig,
) -> RootFnApply:
    def root_fn(
        params: ActorCriticParams,
        observation: chex.ArrayTree,
        env_state: chex.ArrayTree,
        rng_key: chex.PRNGKey,
    ) -> mctx.RootFnOutput:

        sample_key, noise_key = jax.random.split(rng_key, 2)

        pi = actor_apply_fn(params.actor_params, observation)
        value = critic_apply_fn(params.critic_params, observation)
        batch_size = value.shape[0]
        # Sample actions for the root node.
        sampled_actions = pi.sample(seed=sample_key, sample_shape=config.system.num_samples)
        chex.assert_shape(
            sampled_actions, (config.system.num_samples, batch_size, config.system.action_dim)
        )
        # Swap axes to have (batch_size, num_samples, action_dim)
        sampled_actions = jnp.swapaxes(sampled_actions, 0, 1)
        chex.assert_shape(
            sampled_actions, (batch_size, config.system.num_samples, config.system.action_dim)
        )
        # Add noise to the root sampled actions.
        if config.system.root_exploration_fraction != 0:
            sampled_actions = add_gaussian_noise(
                noise_key,
                sampled_actions,
                config.system.root_exploration_fraction,
                config.system.action_minimum,
                config.system.action_maximum,
            )
        # Due to sampling from a gaussian, set all actions to have a uniform prior.
        selection_logits = jnp.ones((batch_size, config.system.num_samples))
        # Pack the search tree state.
        search_tree_state = {
            "env_state": env_state,
            "sampled_actions": sampled_actions,
        }

        root_fn_output = mctx.RootFnOutput(
            prior_logits=selection_logits,
            value=value,
            embedding=search_tree_state,
        )

        return root_fn_output

    return root_fn


def make_recurrent_fn(
    environment_step: EnvironmentStep,
    actor_apply_fn: ActorApply,
    critic_apply_fn: CriticApply,
    config: DictConfig,
) -> mctx.RecurrentFn:
    def recurrent_fn(
        params: ActorCriticParams,
        rng_key: chex.PRNGKey,
        action_index: chex.Array,
        search_tree_state: chex.ArrayTree,
    ) -> Tuple[mctx.RecurrentFnOutput, chex.ArrayTree]:

        env_state = search_tree_state["env_state"]
        batch_size = action_index.shape[0]
        batch_indices = jnp.arange(batch_size)
        action = search_tree_state["sampled_actions"][batch_indices, action_index]

        next_env_state, next_timestep = environment_step(env_state, action)

        pi = actor_apply_fn(params.actor_params, next_timestep.observation)
        value = critic_apply_fn(params.critic_params, next_timestep.observation)

        # Sample actions for the next state.
        next_sampled_actions = pi.sample(seed=rng_key, sample_shape=config.system.num_samples)
        chex.assert_shape(
            next_sampled_actions, (config.system.num_samples, batch_size, config.system.action_dim)
        )
        # Swap axes to have (batch_size, num_samples, action_dim)
        next_sampled_actions = jnp.swapaxes(next_sampled_actions, 0, 1)
        chex.assert_shape(
            next_sampled_actions, (batch_size, config.system.num_samples, config.system.action_dim)
        )
        # Due to sampling from a gaussian, set all actions to have a uniform prior.
        selection_logits = jnp.ones((batch_size, config.system.num_samples))

        # We currently check if the next timestep is
        # truncated by checking if the discount is not zero but the step type is last.
        # Currently, we set the tree search discount to be zero if truncated due to
        # us not being sure how the environment will handle the next state.
        # However, the value that is predicted by the critic network is still used.
        truncated = (next_timestep.last() & (next_timestep.discount != 0.0)).astype(jnp.float32)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=next_timestep.reward,
            discount=next_timestep.discount * config.system.gamma * (1 - truncated),
            prior_logits=selection_logits,
            value=next_timestep.discount * value,
        )

        # Pack the search tree state.
        next_search_tree_state = {
            "env_state": next_env_state,
            "sampled_actions": next_sampled_actions,
        }

        return recurrent_fn_output, next_search_tree_state

    return recurrent_fn


def make_sampled_search_apply_fn(
    config: DictConfig,
    model_recurrent_fn: mctx.RecurrentFn,
) -> SearchApply:

    search_method = parse_search_method(config)
    mctx_search_apply_fn = functools.partial(
        search_method,
        recurrent_fn=model_recurrent_fn,
        num_simulations=config.system.num_simulations,
        max_depth=config.system.max_depth,
        **config.system.search_method_kwargs,
    )

    def search_apply_fn(
        params: ActorCriticParams,
        key: chex.PRNGKey,
        root: mctx.RootFnOutput,
    ) -> mctx.PolicyOutput:
        """Apply the search function."""
        root_sampled_actions = root.embedding["sampled_actions"]
        search_output = mctx_search_apply_fn(params, key, root)
        action_index = search_output.action
        batch_indices = jnp.arange(action_index.shape[0])
        action = root_sampled_actions[batch_indices, action_index]
        chex.assert_shape(action, (action_index.shape[0], config.system.action_dim))
        return search_output.replace(action=action)

    return search_apply_fn


def get_warmup_fn(
    env: Environment,
    params: ActorCriticParams,
    apply_fns: Tuple[ActorApply, CriticApply, RootFnApply, SearchApply],
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:

    _, _, root_fn, search_apply_fn = apply_fns

    def warmup(
        env_states: LogEnvState, timesteps: TimeStep, buffer_states: BufferState, keys: chex.PRNGKey
    ) -> Tuple[LogEnvState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, chex.PRNGKey], SampledExItTransition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, root_key, policy_key = jax.random.split(key, num=3)
            root = root_fn(params, last_timestep.observation, env_state.env_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action
            search_policy = search_output.action_weights
            search_value = search_output.search_tree.node_values[:, mctx.Tree.ROOT_INDEX]
            root_sampled_actions = root.embedding["sampled_actions"]

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = SampledExItTransition(
                done,
                action,
                root_sampled_actions,
                timestep.reward,
                search_value,
                search_policy,
                last_timestep.observation,
                info,
            )

            return (env_state, timestep, key), transition

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
    apply_fns: Tuple[ActorApply, CriticApply, RootFnApply, SearchApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[ZLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn, root_fn, search_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(learner_state: ZLearnerState, _: Any) -> Tuple[ZLearnerState, Tuple]:
        """A single update of the network."""

        def _env_step(
            learner_state: ZLearnerState, _: Any
        ) -> Tuple[ZLearnerState, SampledExItTransition]:
            """Step the environment."""
            params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, root_key, policy_key = jax.random.split(key, num=3)
            root = root_fn(params, last_timestep.observation, env_state.env_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action
            search_policy = search_output.action_weights
            search_value = search_output.search_tree.node_values[:, mctx.Tree.ROOT_INDEX]
            root_sampled_actions = root.embedding["sampled_actions"]

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = SampledExItTransition(
                done,
                action,
                root_sampled_actions,
                timestep.reward,
                search_value,
                search_policy,
                last_timestep.observation,
                info,
            )
            learner_state = ZLearnerState(
                params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

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
                actor_params: FrozenDict,
                sequence: SampledExItTransition,
                rng_key: chex.PRNGKey,
            ) -> Tuple:
                """Calculate the actor loss."""
                # RERUN NETWORK
                sequence = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), sequence)
                actor_policy = actor_apply_fn(actor_params, sequence.obs)

                # CALCULATE LOSS
                log_prob = jax.vmap(actor_policy.log_prob, in_axes=1, out_axes=1)(
                    sequence.sampled_actions
                )
                actor_loss = -jnp.sum(log_prob * sequence.search_policy, -1).mean()
                entropy = actor_policy.entropy(seed=rng_key).mean()

                total_loss_actor = actor_loss - config.system.ent_coef * entropy
                loss_info = {
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return total_loss_actor, loss_info

            def _critic_loss_fn(
                critic_params: FrozenDict,
                sequence: SampledExItTransition,
            ) -> Tuple:
                """Calculate the critic loss."""
                # RERUN NETWORK
                value = critic_apply_fn(critic_params, sequence.obs)[:, :-1]

                # COMPUTE TARGETS
                _, targets = batch_truncated_generalized_advantage_estimation(
                    sequence.reward[:, :-1],
                    (1 - sequence.done)[:, :-1] * config.system.gamma,
                    config.system.gae_lambda,
                    sequence.search_value,
                )

                # CALCULATE VALUE LOSS
                value_loss = rlax.l2_loss(value, targets).mean()

                critic_total_loss = config.system.vf_coef * value_loss
                loss_info = {
                    "value_loss": value_loss,
                }
                return critic_total_loss, loss_info

            params, opt_states, buffer_state, key = update_state

            key, sample_key, actor_key = jax.random.split(key, num=3)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: SampledExItTransition = sequence_sample.experience

            # CALCULATE ACTOR LOSS
            actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(params.actor_params, sequence, actor_key)

            # CALCULATE CRITIC LOSS
            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(params.critic_params, sequence)

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

            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            # pmean over devices.
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )

            # UPDATE ACTOR PARAMS AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

            # UPDATE CRITIC PARAMS AND OPTIMISER STATE
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
            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (params, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key = update_state
        learner_state = ZLearnerState(
            params, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: ZLearnerState) -> ExperimentOutput[ZLearnerState]:
        """Learner function."""

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


def parse_search_method(config: DictConfig) -> Any:
    """Parse search method from config."""
    if config.system.search_method.lower() == "muzero":
        search_method = mctx.muzero_policy
    elif config.system.search_method.lower() == "gumbel":
        search_method = mctx.gumbel_muzero_policy
    else:
        raise ValueError(f"Search method {config.system.search_method} not supported.")

    return search_method


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig, model_env: Environment
) -> Tuple[LearnerFn[ZLearnerState], RootFnApply, SearchApply, ZLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions.
    action_dim = int(env.action_spec().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_spec().minimum)
    config.system.action_maximum = float(env.action_spec().maximum)

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)

    actor_lr = make_learning_rate(
        config.system.actor_lr,
        config,
        config.system.epochs,
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr,
        config,
        config.system.epochs,
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    root_fn = make_root_fn(actor_network_apply_fn, critic_network_apply_fn, config)
    environment_model_step = jax.vmap(model_env.step)
    model_recurrent_fn = make_recurrent_fn(
        environment_model_step, actor_network_apply_fn, critic_network_apply_fn, config
    )
    search_apply_fn = make_sampled_search_apply_fn(config, model_recurrent_fn)

    # Pack apply and update functions.
    apply_fns = (
        actor_network_apply_fn,
        critic_network_apply_fn,
        root_fn,
        search_apply_fn,
    )
    update_fns = (actor_optim.update, critic_optim.update)

    # Create replay buffer
    dummy_transition = SampledExItTransition(
        done=jnp.array(False),
        action=jnp.zeros(action_dim, dtype=jnp.float32),
        sampled_actions=jnp.zeros((config.system.num_samples, action_dim), dtype=jnp.float32),
        reward=jnp.array(0.0),
        search_value=jnp.array(0.0),
        search_policy=jnp.zeros((config.system.num_samples,), dtype=jnp.float32),
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
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
    buffer_states = buffer_fn.init(dummy_transition)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, apply_fns, buffer_fn.add, config)
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
        restored_params, _ = loaded_checkpoint.restore_params()
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
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
    init_learner_state = ZLearnerState(
        params, opt_states, buffer_states, step_keys, env_states, timesteps
    )

    return learn, root_fn, search_apply_fn, init_learner_state


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
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, root_fn, search_apply_fn, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config, eval_env
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = search_evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        search_apply_fn=search_apply_fn,
        root_fn=root_fn,
        params=learner_state.params,
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
    best_params = unreplicate_batch_dim(learner_state.params)
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
            learner_output.learner_state.params
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
            # Save checkpoint of learner state
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
    config_path="../../configs", config_name="default_ff_sampled_az.yaml", version_base="1.2"
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}Sampled AlphaZero experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
