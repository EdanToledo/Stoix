"""Implementation of the IMPALA algorithm using the Sebulba architecture.

This implementation follows the paper "IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures" (https://arxiv.org/abs/1802.01561).

The implementation uses rlax's V-trace implementation and follows patterns from
DeepMind's Acme library for correctness.

Key features:
- V-trace for off-policy correction (using rlax)
- Multi-threaded actors for parallel data collection
- Asynchronous learning with a central learner
- Feed-forward policy network

Implementation Notes:
- Uses rlax.vtrace_td_error_and_advantage for V-trace computation
- Uses separate clipping thresholds for value (ρ̄) and policy gradient (c̄)
- Uses λ=1 for V-trace as specified in the paper
- Follows paper's trajectory length of 20 steps
- Uses importance sampling for off-policy correction
- Supports multiple actors running in parallel
- Uses a pipeline for efficient actor-learner communication
"""

import copy
import queue
import threading
import time
import warnings
from collections import defaultdict
from queue import Queue
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
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CoreLearnerState,
    CriticApply,
    Observation,
    SebulbaExperimentOutput,
    SebulbaLearnerFn,
)
from stoix.evaluator import get_distribution_act_fn, get_sebulba_eval_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.impala.impala_types import (
    ActionArray,
    ImpalaTransition,
    LogProbArray,
    ValueArray,
)
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvFactory
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.sebulba_utils import (
    OnPolicyPipeline,
    ParamsSource,
    RecordTimeTo,
    ThreadLifetime,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_act_fn(
    apply_fns: Tuple[ActorApply, CriticApply]
) -> Callable[
    [ActorCriticParams, Observation, chex.PRNGKey], Tuple[ActionArray, ValueArray, LogProbArray]
]:
    """Get the act function that is used by the actor threads.

    Args:
        apply_fns: Tuple of actor and critic network apply functions

    Returns:
        Function that takes parameters, observation, and RNG key and returns
        action, value estimate, and log probability
    """
    actor_apply_fn, critic_apply_fn = apply_fns

    def actor_fn(
        params: ActorCriticParams, observation: Observation, rng_key: chex.PRNGKey
    ) -> Tuple[ActionArray, ValueArray, LogProbArray]:
        """Get the action, value and log_prob from the actor and critic networks.

        Args:
            params: Actor and critic network parameters
            observation: Environment observation
            rng_key: Random number generator key

        Returns:
            Tuple of (action, value estimate, log probability)
        """
        rng_key, policy_key = jax.random.split(rng_key)
        pi = actor_apply_fn(params.actor_params, observation)
        value = critic_apply_fn(params.critic_params, observation)
        action = pi.sample(seed=policy_key)
        log_prob = pi.log_prob(action)
        return action, value, log_prob

    return actor_fn


def get_rollout_fn(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    params_source: ParamsSource,
    pipeline: OnPolicyPipeline,
    apply_fns: Tuple[ActorApply, CriticApply],
    config: DictConfig,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
) -> Callable[[chex.PRNGKey], None]:
    """Get the rollout function that is used by the actor threads."""
    # Unpack and set up the functions
    act_fn = get_act_fn(apply_fns)
    act_fn = jax.jit(act_fn, device=actor_device)
    cpu = jax.devices("cpu")[0]
    move_to_device = lambda tree: jax.tree.map(lambda x: jax.device_put(x, actor_device), tree)
    split_key_fn = jax.jit(jax.random.split, device=actor_device)
    # Build the environments
    envs = env_factory(config.arch.actor.num_envs_per_actor)

    # Create the rollout function
    def rollout_fn(rng_key: chex.PRNGKey) -> None:
        # Ensure all computation is on the actor device
        with jax.default_device(actor_device):
            # Reset the environment
            timestep = envs.reset(seed=seeds)

            # Loop until the thread is stopped
            while not thread_lifetime.should_stop():
                # Create the list to store transitions
                traj: List[ImpalaTransition] = []
                # Create the dictionary to store timings for metrics
                actor_timings_dict: Dict[str, List[float]] = defaultdict(list)
                episode_metrics: List[Dict[str, List[float]]] = []
                # Rollout the environment
                with RecordTimeTo(actor_timings_dict["single_rollout_time"]):
                    # Loop until the rollout length is reached
                    for _ in range(config.system.rollout_length):
                        # Get the latest parameters from the source
                        with RecordTimeTo(actor_timings_dict["get_params_time"]):
                            params = params_source.get()

                        # Move the environment data to the actor device
                        cached_obs = move_to_device(timestep.observation)

                        # Run the actor and critic networks to get the action, value and log_prob
                        with RecordTimeTo(actor_timings_dict["compute_action_time"]):
                            rng_key, policy_key = split_key_fn(rng_key)
                            action, value, log_prob = act_fn(params, cached_obs, policy_key)

                        # Move the action to the CPU
                        action_cpu = np.asarray(jax.device_put(action, cpu))

                        # Step the environment
                        with RecordTimeTo(actor_timings_dict["env_step_time"]):
                            timestep = envs.step(action_cpu)

                        # Get the next dones and truncation flags
                        dones = np.logical_and(
                            np.asarray(timestep.last()), np.asarray(timestep.discount == 0.0)
                        )
                        trunc = np.logical_and(
                            np.asarray(timestep.last()), np.asarray(timestep.discount == 1.0)
                        )
                        cached_next_dones = move_to_device(dones)
                        cached_next_trunc = move_to_device(trunc)

                        # Append ImpalaTransition to the trajectory list
                        reward = timestep.reward
                        metrics = timestep.extras["metrics"]

                        # Store behavior policy log prob for importance sampling
                        traj.append(
                            ImpalaTransition(
                                cached_next_dones,
                                cached_next_trunc,
                                action,
                                value,
                                reward,
                                log_prob,
                                cached_obs,
                                metrics,
                            )
                        )
                        episode_metrics.append(metrics)

                # Send the trajectory to the pipeline
                with RecordTimeTo(actor_timings_dict["rollout_put_time"]):
                    try:
                        pipeline.put(traj, timestep, actor_timings_dict, episode_metrics)
                    except queue.Full:
                        warnings.warn(
                            "Waited too long to add to the rollout queue, killing the actor thread",
                            stacklevel=2,
                        )
                        break

            # Close the environments
            envs.close()

    return rollout_fn


def get_actor_thread(
    env_factory: EnvFactory,
    actor_device: jax.Device,
    params_source: ParamsSource,
    pipeline: OnPolicyPipeline,
    apply_fns: Tuple[ActorApply, CriticApply],
    rng_key: chex.PRNGKey,
    config: DictConfig,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
    name: str,
) -> threading.Thread:
    """Get the actor thread that once started will collect data from the
    environment and send it to the pipeline."""
    rng_key = jax.device_put(rng_key, actor_device)

    rollout_fn = get_rollout_fn(
        env_factory,
        actor_device,
        params_source,
        pipeline,
        apply_fns,
        config,
        seeds,
        thread_lifetime,
    )

    actor = threading.Thread(
        target=rollout_fn,
        args=(rng_key,),
        name=name,
    )

    return actor


def get_learner_step_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> SebulbaLearnerFn[CoreLearnerState, ImpalaTransition]:
    """Get the learner update function for IMPALA.

    This function creates the main learning update for IMPALA, which includes:
    1. Computing V-trace targets and advantages using importance sampling
    2. Updating the policy using V-trace advantages
    3. Updating the value function using V-trace targets

    The learner processes entire trajectory batches as they arrive from actors,
    without using minibatches or multiple epochs (unlike PPO).

    Args:
        apply_fns: Tuple of actor and critic network apply functions
        update_fns: Tuple of actor and critic optimizer update functions
        config: Configuration dictionary

    Returns:
        Function that performs a single learner update step
    """
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: CoreLearnerState, traj_batch: ImpalaTransition
    ) -> Tuple[CoreLearnerState, Dict[str, Array]]:
        """Perform a single update step using V-trace.

        Args:
            learner_state: Current state of the learner
            traj_batch: Batch of transitions from actors

        Returns:
            Updated learner state and dictionary of metrics
        """
        params, opt_states, key, last_timestep = learner_state

        o_tm1 = traj_batch.obs
        a_tm1 = traj_batch.action
        behavior_log_prob_tm1 = traj_batch.log_prob
        r_t = traj_batch.reward
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        o_last_t = last_timestep.observation

        # Concatenate the last observation with the rest of the observations
        all_obs = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[None, ...]], axis=0), o_tm1, o_last_t
        )

        # Calculate importance sampling ratios
        pi_tm1 = actor_apply_fn(params.actor_params, o_tm1)
        log_prob_tm1 = pi_tm1.log_prob(a_tm1)
        rho_tm1 = jnp.exp(log_prob_tm1 - behavior_log_prob_tm1)

        # Normalize rewards as in the paper
        if config.system.normalize_rewards:
            # Compute mean and std across all parallel envs and timesteps
            r_mean = jnp.mean(r_t)
            r_std = jnp.std(r_t)
            # Normalize and scale rewards
            r_t = config.system.reward_scale * (r_t - r_mean) / (r_std + config.system.reward_eps)

        # Add reward normalization stats to metrics
        extra_metrics = {
            "reward_mean": r_mean if config.system.normalize_rewards else jnp.mean(r_t),
            "reward_std": r_std if config.system.normalize_rewards else jnp.std(r_t),
        }

        def _critic_loss_fn(
            critic_params: FrozenDict,
            all_obs: Array,
            r_t: Array,
            d_t: Array,
            rho_tm1: Array,
        ) -> Tuple[Array, Dict[str, Array]]:
            """Calculate the critic loss using V-trace targets.

            Following IMPALA paper's value function update using V-trace targets.
            The loss is based on squared TD errors from V-trace.

            Args:
                critic_params: Critic network parameters
                all_obs: All observations in sequence
                r_t: Reward sequence
                d_t: Discount sequence (1-done, scaled by gamma)
                rho_tm1: Importance sampling ratios

            Returns:
                Total loss and dictionary of loss components
            """
            values = critic_apply_fn(critic_params, all_obs)
            v_tm1 = values[:-1]
            v_t = values[1:]
            # Following IMPALA paper's V-trace implementation
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
                config.system.vtrace_lambda,  # Paper uses λ=1
                config.system.clip_rho_threshold,  # ρ̄ for value
                config.system.clip_pg_rho_threshold,  # c̄ for policy gradient
            )
            value_loss = jnp.square(vtrace_outputs.errors).mean()

            total_loss = config.system.vf_coef * value_loss
            loss_info = {
                "value_loss": value_loss,
                "q_estimate": vtrace_outputs.q_estimate,
                "pg_advantage": vtrace_outputs.pg_advantage,
            }
            return total_loss, loss_info

        def _actor_loss_fn(
            actor_params: FrozenDict,
            o_tm1: Array,
            a_tm1: Array,
            pg_advantage: Array,
        ) -> Tuple[Array, Dict[str, Array]]:
            """Calculate the actor loss using importance sampling.

            Following IMPALA paper's policy gradient with V-trace advantages.
            The policy gradient is calculated with capped importance sampling weights
            to reduce variance while maintaining a valid gradient estimator.

            Args:
                actor_params: Actor network parameters
                o_tm1: Observations
                a_tm1: Actions
                pg_advantage: Policy gradient advantages from V-trace

            Returns:
                Total loss and dictionary of loss components
            """
            actor_policy = actor_apply_fn(actor_params, o_tm1)
            log_prob = actor_policy.log_prob(a_tm1)

            # Policy gradient loss with V-trace advantages
            policy_loss = -(pg_advantage * log_prob).mean()

            # Add entropy bonus for exploration
            entropy = actor_policy.entropy().mean()
            total_loss = policy_loss - config.system.ent_coef * entropy

            loss_info = {
                "actor_loss": policy_loss,
                "entropy": entropy,
            }
            return total_loss, loss_info

        # CALCULATE CRITIC LOSS
        critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
        critic_grads, critic_loss_info = critic_grad_fn(
            params.critic_params, all_obs, r_t, d_t, rho_tm1
        )

        # Extract pg_advantage from critic_loss_info
        pg_advantage = critic_loss_info["pg_advantage"]
        # CALCULATE ACTOR LOSS
        actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
        actor_grads, actor_loss_info = actor_grad_fn(
            params.actor_params, o_tm1, a_tm1, pg_advantage
        )

        # pmean over devices.
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="device"
        )
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
        loss_info = {**actor_loss_info, **critic_loss_info, **extra_metrics}

        # Update learner state
        learner_state = CoreLearnerState(new_params, new_opt_state, key, last_timestep)
        return learner_state, loss_info

    def learner_step_fn(
        learner_state: CoreLearnerState, traj_batch: ImpalaTransition
    ) -> SebulbaExperimentOutput[CoreLearnerState]:
        """A single step of the learner.

        Args:
            learner_state: Current state of the learner
            traj_batch: Batch of transitions from actors

        Returns:
            Updated learner state and metrics
        """
        learner_state, loss_info = _update_step(learner_state, traj_batch)

        return SebulbaExperimentOutput(
            learner_state=learner_state,
            train_metrics=loss_info,
        )

    return learner_step_fn


def get_learner_rollout_fn(
    learn_step: SebulbaLearnerFn[CoreLearnerState, ImpalaTransition],
    config: DictConfig,
    eval_queue: Queue,
    pipeline: OnPolicyPipeline,
    params_sources: Sequence[ParamsSource],
) -> Callable[[CoreLearnerState], None]:
    """Get the learner rollout function that is used by the learner thread to update the networks.
    This function is what is actually run by the learner thread. It gets the data from the pipeline
    and uses the learner update function to update the networks. It then sends these intermediate
    network parameters to a queue for evaluation."""

    def learner_rollout(learner_state: CoreLearnerState) -> None:
        # Loop for the total number of evaluations selected to be performed.
        for _ in range(config.arch.num_evaluation):
            # Create the lists to store metrics and timings for this learning iteration.
            metrics: List[Tuple[Dict, Dict]] = []
            actor_timings: List[Dict] = []
            learner_timings: Dict[str, List[float]] = defaultdict(list)
            q_sizes: List[int] = []
            with RecordTimeTo(learner_timings["learner_time_per_eval"]):
                # Loop for the number of updates per evaluation
                for _ in range(config.arch.num_updates_per_eval):
                    # Get the trajectory batch from the pipeline
                    # This is blocking so it will wait until the pipeline has data.
                    with RecordTimeTo(learner_timings["rollout_get_time"]):
                        (
                            traj_batch,
                            timestep,
                            actor_times,
                            episode_metrics,
                        ) = pipeline.get(  # type: ignore
                            block=True
                        )
                    # We then replace the timestep in the learner state with the latest timestep
                    # This means the learner has access to the entire trajectory as well as
                    # an additional timestep which it can use to bootstrap.
                    learner_state = learner_state._replace(timestep=timestep)
                    # We then call the update function to update the networks
                    with RecordTimeTo(learner_timings["learner_step_time"]):
                        learner_state, train_metrics = learn_step(learner_state, traj_batch)

                    # We store the metrics and timings for this update
                    metrics.append((episode_metrics, train_metrics))
                    actor_timings.append(actor_times)
                    q_sizes.append(pipeline.qsize())

                    # After the update we need to update the params sources with the new params
                    unreplicated_params = unreplicate(learner_state.params)
                    # We loop over all params sources and update them with the new params
                    # This is so that all the actors can get the latest params
                    for source in params_sources:
                        source.update(unreplicated_params)

            # We then pass all the environment metrics, training metrics, current learner state
            # and timings to the evaluation queue. This is so the evaluator correctly evaluates
            # the performance of the networks at this point in time.
            episode_metrics, train_metrics = jax.tree.map(lambda *x: np.asarray(x), *metrics)
            actor_timings = jax.tree.map(lambda *x: np.mean(x), *actor_timings)
            timing_dict = actor_timings | learner_timings
            timing_dict["pipeline_qsize"] = q_sizes
            timing_dict = jax.tree.map(np.mean, timing_dict, is_leaf=lambda x: isinstance(x, list))
            try:
                # We add a timeout mainly for sanity checks
                # If the queue is full for more than 60 seconds we kill the learner thread
                # This should never happen
                eval_queue.put(
                    (episode_metrics, train_metrics, learner_state, timing_dict), timeout=60
                )
            except queue.Full:
                warnings.warn(
                    "Waited too long to add to the evaluation queue, killing the learner thread. "
                    "This should not happen.",
                    stacklevel=2,
                )
                break

    return learner_rollout


def get_learner_thread(
    learn: SebulbaLearnerFn[CoreLearnerState, ImpalaTransition],
    learner_state: CoreLearnerState,
    config: DictConfig,
    eval_queue: Queue,
    pipeline: OnPolicyPipeline,
    params_sources: Sequence[ParamsSource],
) -> threading.Thread:
    """Get the learner thread that is used to update the networks."""

    learner_rollout_fn = get_learner_rollout_fn(learn, config, eval_queue, pipeline, params_sources)

    learner_thread = threading.Thread(
        target=learner_rollout_fn,
        args=(learner_state,),
        name="Learner",
    )

    return learner_thread


def learner_setup(
    env_factory: EnvFactory,
    keys: chex.Array,
    learner_devices: Sequence[jax.Device],
    config: DictConfig,
) -> Tuple[
    SebulbaLearnerFn[CoreLearnerState, ImpalaTransition],
    Tuple[ActorApply, CriticApply],
    CoreLearnerState,
]:
    """Setup for the learner state and networks."""

    # Create a single environment just to get the observation and action specs.
    env = env_factory(num_envs=1)
    # Get number/dimension of actions.
    num_actions = int(env.action_spec().num_values)
    config.system.action_dim = num_actions
    example_obs = env.observation_spec().generate_value()
    env.close()

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)
    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.rmsprop(config.system.actor_lr, eps=0.01),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.rmsprop(config.system.critic_lr, eps=0.01),
    )

    # Initialise observation
    init_x = example_obs
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    # Extract apply functions.
    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn_step = get_learner_step_fn(apply_fns, update_fns, config)
    learn_step = jax.pmap(learn_step, axis_name="device")

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

    # Define params to be replicated across learner devices.
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate across learner devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=learner_devices)

    # Initialise learner state.
    params, opt_states = replicate_learner
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, len(learner_devices))
    init_learner_state = CoreLearnerState(params, opt_states, step_keys, None)

    return learn_step, apply_fns, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Get the learner and actor devices
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(
        global_devices
    ), "Local and global devices must be the same for now. We dont support multihost just yet"
    # Extract the actor and learner devices
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
    actor_devices = [local_devices[device_id] for device_id in config.arch.actor.device_ids]
    local_learner_devices = [
        local_devices[device_id] for device_id in config.arch.learner.device_ids
    ]
    # For evaluation we simply use the first learner device
    evaluator_device = local_learner_devices[0]
    print(f"{Fore.BLUE}{Style.BRIGHT}Actors devices: {actor_devices}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}Learner devices: {local_learner_devices}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Global devices: {global_devices}{Style.RESET_ALL}")
    # Set the number of learning and acting devices in the config
    # useful for keeping track of experimental setup
    config.num_learner_devices = len(local_learner_devices)
    config.num_actor_devices = len(actor_devices)

    # Perform some checks on the config
    # This additionally calculates certains
    # values based on the config
    config = check_total_timesteps(config)

    # Create the environment factory.
    env_factory = environments.make_factory(config)
    assert isinstance(
        env_factory, EnvFactory
    ), "Environment factory must be an instance of EnvFactory"

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )
    np_rng = np.random.default_rng(config.arch.seed)

    # Setup learner.
    learn_step, apply_fns, learner_state = learner_setup(
        env_factory, (key, actor_net_key, critic_net_key), local_learner_devices, config
    )
    actor_apply_fn, _ = apply_fns
    eval_act_fn = get_distribution_act_fn(config, actor_apply_fn)
    # Setup evaluator.
    evaluator, evaluator_envs = get_sebulba_eval_fn(
        env_factory, eval_act_fn, config, np_rng, evaluator_device
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

    # Get initial parameters
    initial_params = unreplicate(learner_state.params)

    # Get the number of steps consumed by the learner per learner step
    steps_per_learner_step = config.system.rollout_length * config.arch.actor.num_envs_per_actor
    # Get the number of steps consumed by the learner per evaluation
    steps_consumed_per_eval = steps_per_learner_step * config.arch.num_updates_per_eval

    # Creating the pipeline
    # First we create the lifetime so we can stop the pipeline when we want
    pipeline_lifetime = ThreadLifetime()
    # Now we create the pipeline
    pipeline = OnPolicyPipeline(
        config.arch.pipeline_queue_size, local_learner_devices, pipeline_lifetime
    )
    # Start the pipeline
    pipeline.start()

    # Create a single lifetime for all the actors and params sources
    actors_lifetime = ThreadLifetime()
    params_sources_lifetime = ThreadLifetime()

    # Create the params sources and actor threads
    params_sources: List[ParamsSource] = []
    actor_threads: List[threading.Thread] = []
    for actor_device in actor_devices:
        # Create 1 params source per actor device as this will be used
        # to pass the params to the actors
        params_source = ParamsSource(initial_params, actor_device, params_sources_lifetime)
        params_source.start()
        params_sources.append(params_source)
        # Now for each device we choose to create multiple actor threads
        for i in range(config.arch.actor.actor_per_device):
            key, actors_key = jax.random.split(key)
            seeds = np_rng.integers(
                np.iinfo(np.int32).max, size=config.arch.actor.num_envs_per_actor
            ).tolist()
            actor_thread = get_actor_thread(
                env_factory,
                actor_device,
                params_source,
                pipeline,
                apply_fns,
                actors_key,
                config,
                seeds,
                actors_lifetime,
                f"Actor-{actor_device}-{i}",
            )
            actor_thread.start()
            actor_threads.append(actor_thread)

    # Create the evaluation queue
    eval_queue: Queue = Queue(maxsize=config.arch.num_evaluation)
    # Create the learner thread
    learner_thread = get_learner_thread(
        learn_step, learner_state, config, eval_queue, pipeline, params_sources
    )
    learner_thread.start()

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(-1e7)
    best_params = initial_params.actor_params
    # This is the main loop, all it does is evaluation and logging.
    # Acting and learning is happening in their own threads.
    # This loop waits for the learner to finish an update before evaluation and logging.
    for eval_step in range(config.arch.num_evaluation):
        # Get the next set of params and metrics from the learner
        episode_metrics, train_metrics, learner_state, timings_dict = eval_queue.get(block=True)

        # Log the metrics and timings
        t = int(steps_consumed_per_eval * (eval_step + 1))
        timings_dict["timestep"] = t
        logger.log(timings_dict, t, eval_step, LogEvent.MISC)

        episode_metrics, ep_completed = get_final_step_metrics(episode_metrics)
        # Calculate steps per second for actor
        # Here we use the number of steps pushed to the pipeline each time
        # and the average time it takes to do a single rollout across
        # all the updates per evaluation
        episode_metrics["steps_per_second"] = (
            steps_per_learner_step / timings_dict["single_rollout_time"]
        )
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)

        train_metrics["learner_step"] = (eval_step + 1) * config.arch.num_updates_per_eval
        train_metrics["sgd_steps_per_second"] = (config.arch.num_updates_per_eval) / timings_dict[
            "learner_time_per_eval"
        ]
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluate the current model and log the metrics
        unreplicated_actor_params = unreplicate(learner_state.params.actor_params)
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = evaluator(unreplicated_actor_params, eval_key)
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_consumed_per_eval * (eval_step + 1),
                unreplicated_learner_state=unreplicate(learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(unreplicated_actor_params)
            max_episode_return = episode_return

    evaluator_envs.close()
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing learner...{Style.RESET_ALL}")
    # Now we stop the learner
    learner_thread.join()

    # First we stop all actors
    actors_lifetime.stop()

    # Now we stop the actors and params sources
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing actors...{Style.RESET_ALL}")
    pipeline.clear()
    for actor in actor_threads:
        # We clear the pipeline before stopping each actor thread
        # since actors can be blocked on the pipeline
        pipeline.clear()
        actor.join()

    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing pipeline...{Style.RESET_ALL}")
    # Stop the pipeline
    pipeline_lifetime.stop()
    pipeline.join()

    print(f"{Fore.MAGENTA}{Style.BRIGHT}Closing params sources...{Style.RESET_ALL}")
    # Stop the params sources
    params_sources_lifetime.stop()
    for param_source in params_sources:
        param_source.join()

    # Measure absolute metric.
    if config.arch.absolute_metric:
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        abs_metric_evaluator, abs_metric_evaluator_envs = get_sebulba_eval_fn(
            env_factory, eval_act_fn, config, np_rng, evaluator_device, eval_multiplier=10
        )
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = abs_metric_evaluator(best_params, eval_key)

        t = int(steps_consumed_per_eval * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/sebulba",
    config_name="default_ff_impala.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    start = time.monotonic()
    eval_performance = run_experiment(cfg)
    end = time.monotonic()
    print(
        f"{Fore.CYAN}{Style.BRIGHT}IMPALA experiment completed in "
        f"{end - start:.2f}s.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
