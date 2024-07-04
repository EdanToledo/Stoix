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
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticParams,
    CriticApply,
    DistributionCriticApply,
    ExperimentOutput,
    LearnerFn,
    LogEnvState,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.networks.inputs import EmbeddingInput
from stoix.systems.search.evaluator import search_evaluator_setup
from stoix.systems.search.search_types import (
    DynamicsApply,
    ExItTransition,
    MZParams,
    RepresentationApply,
    RootFnApply,
    SearchApply,
    ZLearnerState,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    scale_gradient,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import batch_n_step_bootstrapped_returns
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics

tfd = tfp.distributions


def make_root_fn(
    representation_apply_fn: RepresentationApply,
    actor_apply_fn: ActorApply,
    critic_apply_fn: DistributionCriticApply,
    critic_tx_pair: rlax.TxPair,
) -> RootFnApply:
    def root_fn(
        params: MZParams,
        observation: chex.ArrayTree,
        _: chex.ArrayTree,  # This is the state of the environment and unused in MuZero
        rng_key: chex.PRNGKey,
    ) -> mctx.RootFnOutput:
        observation_embedding = representation_apply_fn(params.world_model_params, observation)

        pi = actor_apply_fn(params.prediction_params.actor_params, observation_embedding)
        value_dist = critic_apply_fn(params.prediction_params.critic_params, observation_embedding)
        value = critic_tx_pair.apply_inv(value_dist.probs)
        logits = pi.logits

        root_fn_output = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=observation_embedding,
        )

        return root_fn_output

    return root_fn


def make_recurrent_fn(
    dynamics_apply_fn: DynamicsApply,
    actor_apply_fn: ActorApply,
    critic_apply_fn: DistributionCriticApply,
    critic_tx_pair: rlax.TxPair,
    reward_tx_pair: rlax.TxPair,
    config: DictConfig,
) -> mctx.RecurrentFn:
    def recurrent_fn(
        params: MZParams,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        state_embedding: chex.ArrayTree,
    ) -> Tuple[mctx.RecurrentFnOutput, chex.ArrayTree]:

        next_state_embedding, next_reward_dist = dynamics_apply_fn(
            params.world_model_params, state_embedding, action
        )
        next_reward = reward_tx_pair.apply_inv(next_reward_dist.probs)

        pi = actor_apply_fn(params.prediction_params.actor_params, next_state_embedding)
        value_dist = critic_apply_fn(params.prediction_params.critic_params, next_state_embedding)
        value = critic_tx_pair.apply_inv(value_dist.probs)
        logits = pi.logits

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=next_reward,
            discount=jnp.ones_like(next_reward) * config.system.gamma,
            prior_logits=logits,
            value=value,
        )

        return recurrent_fn_output, next_state_embedding

    return recurrent_fn


def get_warmup_fn(
    env: Environment,
    params: MZParams,
    apply_fns: Tuple[
        RepresentationApply, DynamicsApply, ActorApply, CriticApply, RootFnApply, SearchApply
    ],
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:

    _, _, _, _, root_fn, search_apply_fn = apply_fns

    def warmup(
        env_states: LogEnvState, timesteps: TimeStep, buffer_states: BufferState, keys: chex.PRNGKey
    ) -> Tuple[LogEnvState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, chex.PRNGKey], ExItTransition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, root_key, policy_key = jax.random.split(key, num=3)
            root = root_fn(params, last_timestep.observation, env_state.env_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action
            search_policy = search_output.action_weights
            search_value = search_output.search_tree.node_values[:, mctx.Tree.ROOT_INDEX]

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = ExItTransition(
                done,
                action,
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
    apply_fns: Tuple[
        RepresentationApply,
        DynamicsApply,
        ActorApply,
        DistributionCriticApply,
        RootFnApply,
        SearchApply,
    ],
    update_fn: optax.TransformUpdateFn,
    buffer_fns: Tuple[Callable, Callable],
    transform_pairs: Tuple[rlax.TxPair, rlax.TxPair],
    config: DictConfig,
) -> LearnerFn[ZLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    (
        representation_apply_fn,
        dynamics_apply_fn,
        actor_apply_fn,
        critic_apply_fn,
        root_fn,
        search_apply_fn,
    ) = apply_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns
    critic_tx_pair, reward_tx_pair = transform_pairs

    def _update_step(learner_state: ZLearnerState, _: Any) -> Tuple[ZLearnerState, Tuple]:
        """A single update of the network."""

        def _env_step(learner_state: ZLearnerState, _: Any) -> Tuple[ZLearnerState, ExItTransition]:
            """Step the environment."""
            params, opt_state, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, root_key, policy_key = jax.random.split(key, num=3)
            root = root_fn(params, last_timestep.observation, env_state.env_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action
            search_policy = search_output.action_weights
            search_value = search_output.search_tree.node_values[:, mctx.Tree.ROOT_INDEX]

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = ExItTransition(
                done,
                action,
                timestep.reward,
                search_value,
                search_policy,
                last_timestep.observation,
                info,
            )
            learner_state = ZLearnerState(params, opt_state, buffer_state, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )
        params, opt_state, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        # Swap the batch and time axes.
        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            def _loss_fn(
                muzero_params: MZParams,
                sequence: ExItTransition,
            ) -> Tuple:
                """Calculate the total MuZero loss."""

                # Calculate the value targets using n-step bootstrapped returns
                # with the search values
                r_t = sequence.reward[:, :-1]
                d_t = 1.0 - sequence.done.astype(jnp.float32)
                d_t = (d_t * config.system.gamma).astype(jnp.float32)
                d_t = d_t[:, :-1]
                search_values = sequence.search_value[:, 1:]
                value_targets = batch_n_step_bootstrapped_returns(
                    r_t, d_t, search_values, config.system.n_steps
                )

                # Get the state embedding of the first observation of each sequence
                state_embedding = representation_apply_fn(
                    muzero_params.world_model_params, sequence.obs
                )[
                    :, 0
                ]  # B, T=0

                def unroll_fn(
                    carry: Tuple[chex.Array, chex.Array, MZParams, chex.Array],
                    targets: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
                ) -> Tuple[chex.Array, chex.Array]:
                    total_loss, state_embedding, muzero_params, mask = carry
                    action, reward_target, search_policy, value_targets, done = targets
                    actor_policy = actor_apply_fn(
                        muzero_params.prediction_params.actor_params, state_embedding
                    )
                    value_dist = critic_apply_fn(
                        muzero_params.prediction_params.critic_params, state_embedding
                    )
                    state_embedding = scale_gradient(state_embedding, 0.5)
                    next_state_embedding, predicted_reward = dynamics_apply_fn(
                        muzero_params.world_model_params, state_embedding, action
                    )

                    # CALCULATE ACTOR LOSS
                    # We use the KL divergence between the search policy and the actor policy
                    # as the actor loss
                    actor_loss = tfd.Categorical(probs=search_policy).kl_divergence(actor_policy)
                    # We mask the loss past the episode end
                    actor_loss = actor_loss * mask

                    # CALCULATE ENTROPY LOSS
                    entropy_loss = config.system.ent_coef * actor_policy.entropy()
                    # We mask the loss past the episode end
                    entropy_loss = entropy_loss * mask

                    # CALCULATE CRITIC LOSS
                    # Here we mask the target instead of the loss since past the
                    # episode end should be treated as an absorbing state
                    # where the value is 0
                    value_targets = value_targets * mask
                    value_targets = critic_tx_pair.apply(value_targets)
                    value_loss = config.system.vf_coef * optax.softmax_cross_entropy(
                        value_dist.logits, value_targets
                    )

                    # CALCULATE REWARD LOSS
                    # We do the same for the reward loss as we did for the value loss
                    reward_target = reward_target * mask
                    reward_target = reward_tx_pair.apply(reward_target)
                    reward_loss = optax.softmax_cross_entropy(
                        predicted_reward.logits, reward_target
                    )

                    curr_loss = {
                        "actor_loss": actor_loss,
                        "value_loss": value_loss,
                        "reward_loss": reward_loss,
                        "entropy_loss": entropy_loss,
                    }
                    # UPDATE LOSS
                    total_loss = jax.tree_util.tree_map(
                        lambda x, y: x + y.mean(), total_loss, curr_loss
                    )
                    # Update the mask - This is to ensure that the loss is
                    # not updated for any steps after the episode is done
                    mask = mask * (1.0 - done.astype(jnp.float32))
                    return (total_loss, next_state_embedding, muzero_params, mask), None

                targets = (
                    sequence.action[:, :-1],
                    sequence.reward[:, :-1],
                    sequence.search_policy[:, :-1],
                    value_targets,
                    sequence.done[:, :-1],
                )  # B, T

                targets = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), targets)  # T, B
                init_total_loss = {
                    "actor_loss": jnp.array(0.0),
                    "value_loss": jnp.array(0.0),
                    "reward_loss": jnp.array(0.0),
                    "entropy_loss": jnp.array(0.0),
                }
                init_mask = jnp.ones((config.system.batch_size,))
                (losses, _, _, _), _ = jax.lax.scan(
                    unroll_fn, (init_total_loss, state_embedding, muzero_params, init_mask), targets
                )
                # Divide by the number of unrolled steps to ensure a consistent scale
                # across different unroll lengths
                losses = jax.tree_util.tree_map(
                    lambda x: x / (config.system.sample_sequence_length - 1), losses
                )

                total_loss = (
                    losses["actor_loss"]
                    + losses["value_loss"]
                    + losses["reward_loss"]
                    - losses["entropy_loss"]
                )

                return total_loss, losses

            params, opt_state, buffer_state, key = update_state

            key, sample_key = jax.random.split(key)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: ExItTransition = sequence_sample.experience

            # CALCULATE LOSS
            grad_fn = jax.grad(_loss_fn, has_aux=True)
            grads, loss_info = grad_fn(
                params,
                sequence,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="batch")
            # pmean over devices.
            grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="device")

            # UPDATE PARAMS AND OPTIMISER STATE
            updates, new_opt_state = update_fn(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (params, opt_state, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_state, buffer_state, key = update_state
        learner_state = ZLearnerState(
            params, opt_state, buffer_state, key, env_state, last_timestep
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
    env: Environment,
    keys: chex.Array,
    config: DictConfig,
) -> Tuple[LearnerFn[ZLearnerState], RootFnApply, SearchApply, ZLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_spec().num_values)
    config.system.action_dim = num_actions

    # PRNG keys.
    key, wm_network_key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(
        config.network.critic_network.critic_head,
    )

    actor_network = Actor(
        torso=actor_torso, action_head=actor_action_head, input_layer=EmbeddingInput()
    )
    critic_network = Critic(
        torso=critic_torso, critic_head=critic_head, input_layer=EmbeddingInput()
    )

    wm_network = hydra.utils.instantiate(
        config.network.wm_network, action_dim=config.system.action_dim
    )

    lr = make_learning_rate(
        config.system.lr,
        config,
        config.system.epochs,
    )

    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_a = env.action_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    init_a = jax.tree_util.tree_map(lambda x: x[None, ...], init_a)

    # Initialise params params and optimiser state.
    world_model_params = wm_network.init(wm_network_key, init_x, init_a)
    hidden_state_embedding, _ = wm_network.apply(world_model_params, init_x, init_a)
    actor_params = actor_network.init(actor_net_key, hidden_state_embedding)
    critic_params = critic_network.init(critic_net_key, hidden_state_embedding)

    # Pack params.
    prediction_params = ActorCriticParams(actor_params, critic_params)
    params = MZParams(prediction_params, world_model_params)

    # Initialise optimiser state.
    opt_state = optim.init(params)

    # Define apply functions.
    representation_network_apply_fn = functools.partial(
        wm_network.apply, method=wm_network.initial_inference
    )
    dynamics_network_apply_fn = functools.partial(
        wm_network.apply, method=wm_network.recurrent_inference
    )
    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Initialise tx pairs.
    critic_tx_pair = rlax.muzero_pair(
        config.system.critic_vmin,
        config.system.critic_vmax,
        config.system.critic_num_atoms,
        rlax.SIGNED_HYPERBOLIC_PAIR,
    )
    reward_tx_pair = rlax.muzero_pair(
        config.system.reward_vmin,
        config.system.reward_vmax,
        config.system.reward_num_atoms,
        rlax.SIGNED_HYPERBOLIC_PAIR,
    )

    root_fn = make_root_fn(
        representation_network_apply_fn,
        actor_network_apply_fn,
        critic_network_apply_fn,
        critic_tx_pair,
    )
    model_recurrent_fn = make_recurrent_fn(
        dynamics_network_apply_fn,
        actor_network_apply_fn,
        critic_network_apply_fn,
        critic_tx_pair,
        reward_tx_pair,
        config,
    )
    search_method = parse_search_method(config)
    search_apply_fn = functools.partial(
        search_method,
        recurrent_fn=model_recurrent_fn,
        num_simulations=config.system.num_simulations,
        max_depth=config.system.max_depth,
        **config.system.search_method_kwargs,
    )

    # Pack apply and update functions.
    apply_fns = (
        representation_network_apply_fn,
        dynamics_network_apply_fn,
        actor_network_apply_fn,
        critic_network_apply_fn,
        root_fn,
        search_apply_fn,
    )
    update_fns = optim.update
    transform_pairs = (critic_tx_pair, reward_tx_pair)

    # Create replay buffer
    dummy_transition = ExItTransition(
        done=jnp.array(False),
        action=jnp.array(0),
        reward=jnp.array(0.0),
        search_value=jnp.array(0.0),
        search_policy=jnp.zeros((num_actions,)),
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
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, transform_pairs, config)
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
        restored_params, _ = loaded_checkpoint.restore_params(TParams=MZParams)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (params, opt_state, buffer_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_state, buffer_states = replicate_learner
    # Warmup the buffer.
    env_states, timesteps, keys, buffer_states = warmup(
        env_states, timesteps, buffer_states, warmup_keys
    )
    init_learner_state = ZLearnerState(
        params, opt_state, buffer_states, step_keys, env_states, timesteps
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
    key, key_e, wm_key, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=5
    )

    # Setup learner.
    learn, root_fn, search_apply_fn, learner_state = learner_setup(
        env, (key, wm_key, actor_net_key, critic_net_key), config
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


@hydra.main(config_path="../../configs", config_name="default_ff_mz.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}MuZero experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
