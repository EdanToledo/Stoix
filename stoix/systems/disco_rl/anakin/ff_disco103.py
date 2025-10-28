import copy
import time
from typing import Any, Callable, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from disco_rl import types as disco_types
from disco_rl.update_rules import disco as disco_rule
from flax.core.frozen_dict import FrozenDict
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, get_final_step_metrics
from tensorflow_probability.substrates.jax.distributions import Categorical

from stoix.base_types import AnakinExperimentOutput, LearnerFn
from stoix.evaluator import evaluator_setup
from stoix.networks.specialised.disco103 import DiscoAgentNetwork
from stoix.systems.disco_rl.disco_rl_types import (
    AgentOutput,
    DiscoLearnerState,
    DiscoTransition,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.download import get_or_create_file
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_learner_fn(
    env: Environment,
    agent_apply_fn: Callable,
    agent_update_fn: optax.TransformUpdateFn,
    meta_update_rule: disco_rule.DiscoUpdateRule,
    config: DictConfig,
) -> LearnerFn[DiscoLearnerState]:
    """Get the learner function."""

    def _update_step(learner_state: DiscoLearnerState, _: Any) -> Tuple[DiscoLearnerState, Tuple]:
        def _env_step(
            learner_state: DiscoLearnerState, _: Any
        ) -> Tuple[DiscoLearnerState, DiscoTransition]:
            """Step the environment."""
            params, _, key, env_state, last_timestep, _, _ = learner_state

            # GET OBSERVATION
            observation = last_timestep.observation

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            agent_output: AgentOutput = agent_apply_fn(params, observation)
            pi = Categorical(logits=agent_output.logits)
            action = pi.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = DiscoTransition(
                done,
                truncated,
                action,
                timestep.reward,
                last_timestep.observation,
                info,
                agent_output,
            )
            # Replace the learner state with the new environment state and timestep.
            learner_state = learner_state._replace(
                key=key,
                env_state=env_state,
                timestep=timestep,
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # UNPACK LEARNER STATE
        params, opt_states, key, _, _, meta_params, meta_state = learner_state

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, minibatch_traj: chex.ArrayTree) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE
                params, opt_states, meta_params, meta_state, key = train_state

                def agent_unroll_fn(
                    params: FrozenDict,
                    unused_state: chex.ArrayTree,
                    observations: chex.ArrayTree,
                    unused_should_reset_mask_fwd: chex.Array,
                ) -> Tuple[chex.ArrayTree, None]:
                    """Since this is a feedforward network, we can just vmap over time"""
                    apply_fn = lambda obs: agent_apply_fn(params, obs)
                    agent_out = jax.vmap(apply_fn)(observations)
                    return agent_out._asdict(), unused_state

                def _agent_loss_fn(
                    params: FrozenDict,
                    meta_params: FrozenDict,
                    minibatch_traj: DiscoTransition,
                    hyperparams: dict,
                    meta_state: disco_types.MetaState,  # type: ignore
                    rng_key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the agent loss."""

                    current_agent_out, _ = agent_unroll_fn(params, None, minibatch_traj.obs, None)
                    behaviour_agent_out = minibatch_traj.agent_out

                    update_rule_inputs = disco_types.UpdateRuleInputs(
                        observations=minibatch_traj.obs,
                        actions=minibatch_traj.action,
                        rewards=minibatch_traj.reward[:-1],
                        is_terminal=minibatch_traj.done[:-1],
                        agent_out=current_agent_out,
                        behaviour_agent_out=behaviour_agent_out._asdict(),
                    )

                    # Compute the loss per step
                    loss_per_step, new_meta_state, logs = meta_update_rule(
                        meta_params,
                        params,
                        None,
                        update_rule_inputs,
                        hyperparams,
                        meta_state,
                        agent_unroll_fn,
                        rng_key,
                        axis_name="device",
                        backprop=False,
                    )
                    # Compute total loss
                    total_loss = jnp.mean(loss_per_step)
                    return total_loss, (new_meta_state, logs)

                key, loss_key = jax.random.split(key)

                # CALCULATE AGENT LOSS
                agent_grad_fn = jax.grad(_agent_loss_fn, has_aux=True)
                agent_grads, (new_meta_state, agent_loss_info) = agent_grad_fn(
                    params,
                    meta_params,
                    minibatch_traj,
                    dict(config.system.disco_hyperparams),
                    meta_state,
                    loss_key,
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                agent_grads, agent_loss_info = jax.lax.pmean(
                    (agent_grads, agent_loss_info),
                    axis_name="batch",
                )
                # pmean over devices.
                agent_grads, agent_loss_info = jax.lax.pmean(
                    (agent_grads, agent_loss_info),
                    axis_name="device",
                )

                # UPDATE AGENT PARAMS AND OPTIMISER STATE
                agent_updates, agent_new_opt_state = agent_update_fn(agent_grads, opt_states)
                agent_new_params = optax.apply_updates(params, agent_updates)

                new_params = agent_new_params
                new_opt_state = agent_new_opt_state

                return (
                    new_params,
                    new_opt_state,
                    meta_params,
                    new_meta_state,
                    key,
                ), agent_loss_info

            (
                params,
                opt_states,
                traj_batch,
                key,
                meta_params,
                meta_state,
            ) = update_state
            key, shuffle_key = jax.random.split(key)

            # RESCALE REWARDS
            traj_batch = traj_batch._replace(
                reward=traj_batch.reward.astype(jnp.float32) * config.system.reward_scale
            )

            # SHUFFLE MINIBATCHES
            batch = traj_batch
            permutation = jax.random.permutation(shuffle_key, config.arch.num_envs)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            reshaped_batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # UPDATE MINIBATCHES
            (params, opt_states, meta_params, meta_state, key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, meta_params, meta_state, key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                key,
                meta_params,
                meta_state,
            )
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, key, meta_params, meta_state)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, key, meta_params, meta_state = update_state
        learner_state = learner_state._replace(
            params=params,
            opt_states=opt_states,
            key=key,
            meta_params=meta_params,
            meta_state=meta_state,
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: DiscoLearnerState,
    ) -> AnakinExperimentOutput[DiscoLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (WrapperState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
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


def unflatten_params(flat_params: chex.ArrayTree) -> chex.ArrayTree:
    params = {}
    for key_wb in flat_params:
        key = "/".join(key_wb.split("/")[:-1])
        params[key] = {
            "b": flat_params[f"{key}/b"],
            "w": flat_params[f"{key}/w"],
        }
    return params


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[DiscoLearnerState], DiscoAgentNetwork, DiscoLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    # PRNG keys.
    key, agent_net_key = keys

    # Create the Disco103 meta-update rule
    disco_rule_kwargs = dict(config.system.disco_rule)
    disco_rule_kwargs["net"] = ConfigDict(disco_rule_kwargs["net"])
    meta_net_input_option = disco_rule.get_input_option()
    disco_rule_kwargs["net"].input_option = meta_net_input_option
    # Instantiate the update rule
    meta_update_rule = disco_rule.DiscoUpdateRule(**disco_rule_kwargs)
    random_update_rule_params, _ = meta_update_rule.init_params(jax.random.PRNGKey(0))
    # Load Meta-Params (disco103 weights)
    print(f"{Fore.GREEN}{Style.BRIGHT}Loading Disco103 meta-parameters...{Style.RESET_ALL}")
    disco_103_fname = "disco_103.npz"
    disco_103_url = f"https://raw.githubusercontent.com/google-deepmind/disco_rl/main/disco_rl/update_rules/weights/{disco_103_fname}"  # noqa: E501
    # Download the weights if not already present
    path = get_or_create_file(
        disco_103_fname, disco_103_url, cache_dir="disco_rl/weights", filetype="npz"
    )
    with open(f"{path}", "rb") as file:
        meta_params = unflatten_params(np.load(file))

    chex.assert_trees_all_equal_shapes_and_dtypes(random_update_rule_params, meta_params)
    print(f"{Fore.GREEN}{Style.BRIGHT}Update rule parameters have the same specs.{Style.RESET_ALL}")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Loaded {len(meta_params) * 2} parameter tensors "
        f"for Disco103.{Style.RESET_ALL}"
    )
    print(f"{Fore.GREEN}{Style.BRIGHT}Disco103 meta-parameters loaded.{Style.RESET_ALL}")

    # Get model spec from update rule
    action_spec = disco_types.ActionSpec(
        shape=(), minimum=0, maximum=num_actions - 1, dtype=jnp.int32
    )
    out_model_spec = meta_update_rule.model_output_spec(action_spec)

    # Define network and optimiser.
    shared_torso = hydra.utils.instantiate(config.network.agent_network.shared_torso)
    logits_head = hydra.utils.instantiate(
        config.network.agent_network.logits_head, output_dim=num_actions
    )
    y_head = hydra.utils.instantiate(
        config.network.agent_network.y_head, output_dim=int(out_model_spec["z"].shape[-1])
    )
    # Action Conditional Torso
    action_conditional_torso = hydra.utils.instantiate(
        config.network.agent_network.action_conditional_torso, num_actions=num_actions
    )
    # Action Conditional Heads
    q_head = hydra.utils.instantiate(
        config.network.agent_network.q_head,
        output_dim=int(out_model_spec["q"].shape[-1]),
    )
    z_head = hydra.utils.instantiate(
        config.network.agent_network.z_head,
        output_dim=int(out_model_spec["z"].shape[-1]),
    )
    aux_pi_head = hydra.utils.instantiate(
        config.network.agent_network.aux_pi_head,
        output_dim=int(out_model_spec["aux_pi"].shape[-1]),
    )

    # Instantiate the agent network
    agent_network = DiscoAgentNetwork(
        shared_torso=shared_torso,
        action_conditional_torso=action_conditional_torso,
        logits_head=logits_head,
        q_head=q_head,
        y_head=y_head,
        z_head=z_head,
        aux_pi_head=aux_pi_head,
    )

    lr = make_learning_rate(
        config.system.lr, config, config.system.epochs, config.system.num_minibatches
    )

    agent_optim = optax.chain(
        optax.clip(config.system.max_abs_update),
        optax.adam(lr),
    )

    # Initialise observation
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise agent params and optimiser state.
    params = agent_network.init(agent_net_key, init_x)
    opt_states = agent_optim.init(params)

    # Init Meta-State
    key, meta_key = jax.random.split(key)
    # The meta_state holds the target network, so we pass the initial agent params
    meta_state = meta_update_rule.init_meta_state(meta_key, params)

    # Get agent apply function
    agent_network_apply_fn = agent_network.apply

    # Pack apply and update functions.
    apply_fns = agent_network_apply_fn
    update_fns = agent_optim.update

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, meta_update_rule, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
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
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    replicate_learner = (params, opt_states, meta_params, meta_state)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, meta_params, meta_state = replicate_learner
    init_learner_state = DiscoLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        meta_params=meta_params,
        meta_state=meta_state,
    )

    return learn, agent_network, init_learner_state


def get_disco_eval_act_fn(
    agent_apply_fn: Callable[
        [FrozenDict, chex.Array],
        AgentOutput,
    ],
) -> Callable:
    """Get the action function for evaluation."""

    def eval_act_fn(params: FrozenDict, obs: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Select action with exploration noise."""
        agent_out = agent_apply_fn(params, obs)
        policy = Categorical(logits=agent_out.logits)
        action = policy.sample(seed=key)
        return action

    return eval_act_fn


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
    key, key_e, agent_net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)

    # Setup learner.
    learn, agent_network, learner_state = learner_setup(env, (key, agent_net_key), config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_disco_eval_act_fn(agent_network.apply),
        params=learner_state.params,
        config=config,
    )

    # Calculate environment steps per evaluation.
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
    best_learner_state = unreplicate_batch_dim(learner_state)
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
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

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
            best_learner_state = copy.deepcopy(unreplicate_batch_dim(learner_state))
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        best_params = best_learner_state.params
        evaluator_output = absolute_metric_evaluator(
            best_params,
            eval_keys,
        )
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
    config_path="../../../configs/default/anakin",
    config_name="default_ff_disco103.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    t0 = time.time()
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Disco103 experiment completed in "
        f"{time.time() - t0:.2f} seconds.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
