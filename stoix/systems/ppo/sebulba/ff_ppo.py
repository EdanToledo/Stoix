import copy
import queue
import threading
import warnings
from collections import defaultdict
from queue import Queue
from typing import Any, Callable, Dict, List, Sequence, Tuple

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
from stoix.systems.ppo.ppo_types import PPOTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvFactory
from stoix.utils.jax_utils import merge_leading_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.sebulba_utils import (
    OnPolicyPipeline,
    ParamsSource,
    RecordTimeTo,
    ThreadLifetime,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_act_fn(
    apply_fns: Tuple[ActorApply, CriticApply]
) -> Callable[
    [ActorCriticParams, Observation, chex.PRNGKey], Tuple[chex.Array, chex.Array, chex.Array]
]:
    """Get the act function that is used by the actor threads."""
    actor_apply_fn, critic_apply_fn = apply_fns

    def actor_fn(
        params: ActorCriticParams, observation: Observation, rng_key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Get the action, value and log_prob from the actor and critic networks."""
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
    envs = env_factory(config.arch.actor.envs_per_actor)

    # Create the rollout function
    def rollout_fn(rng_key: chex.PRNGKey) -> None:
        # Ensure all computation is on the actor device
        with jax.default_device(actor_device):
            # Reset the environment
            timestep = envs.reset(seed=seeds)

            # Loop until the thread is stopped
            while not thread_lifetime.should_stop():
                # Create the list to store transitions
                traj: List[PPOTransition] = []
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

                        # Append PPOTransition to the trajectory list
                        reward = timestep.reward
                        metrics = timestep.extras["metrics"]
                        traj.append(
                            PPOTransition(
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
) -> SebulbaLearnerFn[CoreLearnerState, PPOTransition]:
    """Get the learner update function which is used to update the actor and critic networks.
    This function is used by the learner thread to update the networks."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: CoreLearnerState, traj_batch: PPOTransition
    ) -> Tuple[CoreLearnerState, Dict[str, chex.Array]]:

        # CALCULATE ADVANTAGE
        params, opt_states, key, last_timestep = learner_state
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

        r_t = traj_batch.reward
        v_t = jnp.concatenate([traj_batch.value, last_val[None, ...]], axis=0)
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            v_t,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
            truncation_flags=traj_batch.truncated,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(
                        log_prob, traj_batch.log_prob, gae, config.system.clip_eps
                    )
                    entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # CALCULATE VALUE LOSS
                    value_loss = clipped_value_loss(
                        value, traj_batch.value, targets, config.system.clip_eps
                    )

                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return critic_total_loss, loss_info

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
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
                return (new_params, new_opt_state), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            # Since we shard the envs per actor across the devices
            envs_per_batch = config.arch.actor.envs_per_actor // len(config.arch.learner.device_ids)
            batch_size = config.system.rollout_length * envs_per_batch
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = CoreLearnerState(params, opt_states, key, last_timestep)

        return learner_state, loss_info

    def learner_step_fn(
        learner_state: CoreLearnerState, traj_batch: PPOTransition
    ) -> SebulbaExperimentOutput[CoreLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """

        learner_state, loss_info = _update_step(learner_state, traj_batch)

        return SebulbaExperimentOutput(
            learner_state=learner_state,
            train_metrics=loss_info,
        )

    return learner_step_fn


def get_learner_rollout_fn(
    learn_step: SebulbaLearnerFn[CoreLearnerState, PPOTransition],
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
                with RecordTimeTo(learner_timings["learning_time"]):
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
    learn: SebulbaLearnerFn[CoreLearnerState, PPOTransition],
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
    SebulbaLearnerFn[CoreLearnerState, PPOTransition],
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

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
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

    # Perform some checks on the config
    # This additionally calculates certains
    # values based on the config
    config = check_total_timesteps(config)

    assert (
        config.arch.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate the number of updates per evaluation
    config.arch.num_updates_per_eval = int(config.arch.num_updates // config.arch.num_evaluation)

    # Get the learner and actor devices
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(
        global_devices
    ), "Local and global devices must be the same for now. We dont support multihost just yet"
    # Extract the actor and learner devices
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
    config.num_learning_devices = len(local_learner_devices)
    config.num_actor_actor_devices = len(actor_devices)

    # Calculate the number of envs per actor
    assert (
        config.arch.num_envs == config.arch.total_num_envs
    ), "arch.num_envs must equal arch.total_num_envs for Sebulba architectures"
    # We first simply take the total number of envs and divide by the number of actor devices
    # to get the number of envs per actor device
    num_envs_per_actor_device = config.arch.total_num_envs // len(actor_devices)
    # We then divide this by the number of actors per device to get the number of envs per actor
    num_envs_per_actor = int(num_envs_per_actor_device // config.arch.actor.actor_per_device)
    config.arch.actor.envs_per_actor = num_envs_per_actor

    # We then perform a simple check to ensure that the number of envs per actor is
    # divisible by the number of learner devices. This is because we shard the envs
    # per actor across the learner devices This check is mainly relevant for on-policy
    # algorithms
    assert num_envs_per_actor % len(local_learner_devices) == 0, (
        f"The number of envs per actor must be divisible by the number of learner devices. "
        f"Got {num_envs_per_actor} envs per actor and {len(local_learner_devices)} learner devices"
    )

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

    # Get the number of steps per rollout
    steps_per_rollout = (
        config.system.rollout_length * config.arch.total_num_envs * config.arch.num_updates_per_eval
    )

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
                np.iinfo(np.int32).max, size=config.arch.actor.envs_per_actor
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
        t = int(steps_per_rollout * (eval_step + 1))
        timings_dict["timestep"] = t
        logger.log(timings_dict, t, eval_step, LogEvent.MISC)

        episode_metrics, ep_completed = get_final_step_metrics(episode_metrics)
        episode_metrics["steps_per_second"] = (
            steps_per_rollout / timings_dict["single_rollout_time"]
        )
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)

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
                timestep=steps_per_rollout * (eval_step + 1),
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

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/sebulba",
    config_name="default_ff_ppo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}PPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
