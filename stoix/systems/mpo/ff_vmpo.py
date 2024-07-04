import copy
import time
from typing import Any, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    CriticApply,
    ExperimentOutput,
    LearnerFn,
    OnlineAndTarget,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.mpo.continuous_loss import _MPO_FLOAT_EPSILON
from stoix.systems.mpo.discrete_loss import (
    clip_categorical_mpo_params,
    get_temperature_from_params,
)
from stoix.systems.mpo.mpo_types import (
    CategoricalDualParams,
    SequenceStep,
    VMPOLearnerState,
    VMPOOptStates,
    VMPOParams,
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
    batch_truncated_generalized_advantage_estimation,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[VMPOLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn, dual_update_fn = update_fns

    def _update_step(learner_state: VMPOLearnerState, _: Any) -> Tuple[VMPOLearnerState, Tuple]:
        def _env_step(
            learner_state: VMPOLearnerState, _: Any
        ) -> Tuple[VMPOLearnerState, SequenceStep]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep, learner_step_count = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            # We act with target params in VMPO
            actor_policy = actor_apply_fn(params.actor_params.target, last_timestep.observation)
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

            learner_state = VMPOLearnerState(
                params, opt_states, key, env_state, timestep, learner_step_count
            )
            return learner_state, sequence_step

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, key, env_state, last_timestep, learner_step_count = learner_state

        # Swap the batch and time axes for easier processing.
        traj_batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        chex.assert_tree_shape_prefix(
            traj_batch, (config.arch.num_envs, config.system.rollout_length)
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _actor_loss_fn(
                online_actor_params: FrozenDict,
                dual_params: CategoricalDualParams,
                target_actor_params: FrozenDict,
                advantages: chex.Array,
                sequence: SequenceStep,
            ) -> chex.Array:

                # Remove the last timestep from the sequence.
                sequence = jax.tree_util.tree_map(lambda x: x[:, :-1], sequence)

                # Reshape the sequence to [B*T, ...].
                (sequence, advantages) = jax.tree_util.tree_map(
                    lambda x: merge_leading_dims(x, 2), (sequence, advantages)
                )

                temperature = get_temperature_from_params(dual_params).squeeze()
                alpha = jax.nn.softplus(dual_params.log_alpha).squeeze() + _MPO_FLOAT_EPSILON

                online_actor_policy = actor_apply_fn(online_actor_params, sequence.obs)
                target_actor_policy = actor_apply_fn(target_actor_params, sequence.obs)

                sample_log_probs = online_actor_policy.log_prob(sequence.action)
                temperature_constraint = rlax.LagrangePenalty(
                    temperature, config.system.epsilon, False
                )
                kl = target_actor_policy.kl_divergence(online_actor_policy)
                alpha_constraint = rlax.LagrangePenalty(alpha, config.system.epsilon_policy, False)
                kl_constraints = [(kl, alpha_constraint)]
                # restarting_weights = 1-sequence.done.astype(jnp.float32)

                loss, loss_info = rlax.vmpo_loss(
                    sample_log_probs=sample_log_probs,
                    advantages=advantages,
                    temperature_constraint=temperature_constraint,
                    kl_constraints=kl_constraints,
                    # restarting_weights=restarting_weights
                )

                loss_info = loss_info._asdict()
                loss_info["temperature"] = temperature
                loss_info["alpha"] = alpha
                loss_info["advantages"] = advantages

                return jnp.mean(loss), loss_info

            def _critic_loss_fn(
                online_critic_params: FrozenDict,
                value_target: chex.Array,
                sequence: SequenceStep,
            ) -> chex.Array:

                # Remove the last timestep from the sequence.
                sequence = jax.tree_util.tree_map(lambda x: x[:, :-1], sequence)

                online_v_t = critic_apply_fn(online_critic_params, sequence.obs)  # [B, T]

                td_error = value_target - online_v_t

                v_loss = rlax.l2_loss(td_error).mean()

                loss_info = {
                    "v_loss": v_loss,
                }

                return v_loss, loss_info

            params, opt_states, key, sequence_batch, learner_step_count = update_state

            # Calculate Advantages and Value Target of pi_target
            discount = 1.0 - sequence_batch.done.astype(jnp.float32)
            d_t = (discount * config.system.gamma).astype(jnp.float32)
            r_t = jnp.clip(
                sequence_batch.reward, -config.system.max_abs_reward, config.system.max_abs_reward
            ).astype(jnp.float32)

            online_v_t = critic_apply_fn(params.critic_params, sequence_batch.obs)  # [B, T]

            # We recompute the targets using the latest critic every time
            if config.system.use_n_step_bootstrap:
                value_target = batch_n_step_bootstrapped_returns(
                    r_t[:, :-1],
                    d_t[:, :-1],
                    online_v_t[:, 1:],
                    config.system.n_step_for_sequence_bootstrap,
                )
                advantages = value_target - online_v_t[:, :-1]
            else:
                advantages, value_target = batch_truncated_generalized_advantage_estimation(
                    r_t[:, :-1],
                    d_t[:, :-1],
                    config.system.gae_lambda,
                    online_v_t,
                    time_major=False,
                    truncation_flags=sequence_batch.truncated[:, :-1],
                )

            # CALCULATE ACTOR AND DUAL LOSS
            actor_dual_grad_fn = jax.grad(_actor_loss_fn, argnums=(0, 1), has_aux=True)
            actor_dual_grads, actor_dual_loss_info = actor_dual_grad_fn(
                params.actor_params.online,
                params.dual_params,
                params.actor_params.target,
                advantages,
                sequence_batch,
            )

            # CALCULATE CRITIC LOSS
            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(
                params.critic_params,
                value_target,
                sequence_batch,
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

            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )

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

            # UPDATE CRITIC PARAMS AND OPTIMISER STATE
            critic_updates, critic_new_opt_state = critic_update_fn(
                critic_grads, opt_states.critic_opt_state
            )
            critic_new_online_params = optax.apply_updates(params.critic_params, critic_updates)

            learner_step_count += 1

            # POLYAK UPDATE FOR ACTOR
            new_target_actor_params = optax.periodic_update(
                actor_new_online_params,
                params.actor_params.target,
                learner_step_count,
                config.system.actor_target_period,
            )

            # PACK NEW PARAMS AND OPTIMISER STATE
            actor_new_params = OnlineAndTarget(actor_new_online_params, new_target_actor_params)

            new_params = VMPOParams(actor_new_params, critic_new_online_params, dual_new_params)
            new_opt_state = VMPOOptStates(
                actor_new_opt_state, critic_new_opt_state, dual_new_opt_state
            )

            # PACK LOSS INFO
            loss_info = {
                **actor_dual_loss_info,
                **critic_loss_info,
            }
            return (new_params, new_opt_state, key, sequence_batch, learner_step_count), loss_info

        update_state = (params, opt_states, key, traj_batch, learner_step_count)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, key, traj_batch, learner_step_count = update_state
        learner_state = VMPOLearnerState(
            params, opt_states, key, env_state, last_timestep, learner_step_count
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: VMPOLearnerState) -> ExperimentOutput[VMPOLearnerState]:
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
) -> Tuple[LearnerFn[VMPOLearnerState], Actor, VMPOLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions or action dimension from the environment.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define actor_network, critic_network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=action_dim
    )
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_network_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    critic_network = Critic(torso=critic_network_torso, critic_head=critic_network_head)

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs)
    critic_lr = make_learning_rate(config.system.critic_lr, config, config.system.epochs)

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
    target_actor_params = copy.deepcopy(actor_params)

    # Initialise critic params and optimiser state.
    online_critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(online_critic_params)

    # Initialise VMPO Dual params and optimiser state.
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

    params = VMPOParams(
        OnlineAndTarget(actor_params, target_actor_params),
        online_critic_params,
        dual_params,
    )
    opt_states = VMPOOptStates(actor_opt_state, critic_opt_state, dual_opt_state)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update, dual_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

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
        restored_params, _ = loaded_checkpoint.restore_params(TParams=VMPOParams)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key, num=2)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    learner_step_count = jnp.int32(0)

    replicate_learner = (params, opt_states, learner_step_count)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, learner_step_count = replicate_learner

    init_learner_state = VMPOLearnerState(
        params, opt_states, step_keys, env_states, timesteps, learner_step_count
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
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
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


@hydra.main(config_path="../../configs", config_name="default_ff_vmpo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}V-MPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
