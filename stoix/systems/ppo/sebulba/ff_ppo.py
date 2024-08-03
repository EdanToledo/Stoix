import copy
import threading
import time
from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Sequence, Tuple

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
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    CriticApply,
    EnvFactory,
    ExperimentOutput,
    LearnerFn,
    LearnerState,
    SebulbaLearnerFn,
)
from stoix.evaluator import (
    evaluator_setup,
    get_distribution_act_fn,
    get_sebulba_eval_fn,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.ppo.ppo_types import PPOTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.env_factory import EnvPoolFactory, make_gym_env_factory
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.sebulba_utils import (
    ParamsSource,
    Pipeline,
    RecordTimeTo,
    ThreadLifetime,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_rollout_fn(
    env_builder: EnvFactory,
    actor_device: jax.Device,
    params_source: ParamsSource,
    pipeline: Pipeline,
    apply_fns: Tuple[ActorApply, CriticApply],
    config: DictConfig,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
):

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_apply_fn = jax.jit(actor_apply_fn)
    critic_apply_fn = jax.jit(critic_apply_fn)
    cpu = jax.devices("cpu")[0]
    split_key_fn = jax.jit(jax.random.split)
    envs = env_builder(config.arch.actor.envs_per_actor)
    move_to_device = jax.jit(lambda x: jax.device_put(x, actor_device))

    # Create the rollout function
    def rollout(rng: chex.PRNGKey) -> None:
        with jax.default_device(actor_device):
            # Reset the environment
            # TODO(edan): put seeds in reset
            timestep = envs.reset(seed=seeds)
            next_dones = np.logical_and(
                np.array(timestep.last()), np.array(timestep.discount == 0.0)
            )
            next_trunc = np.logical_and(
                np.array(timestep.last()), np.array(timestep.discount == 1.0)
            )
            # Loop until the thread is stopped
            while not thread_lifetime.should_stop():
                # Create the list to store transitions
                traj: List[PPOTransition] = []
                # Create the dictionary to store timings
                timings_dict: Dict[str, List[float]] = defaultdict(list)

                for _ in range(config.system.rollout_length):
                    with RecordTimeTo(timings_dict["get_params_time"]):
                        params = params_source.get()

                    cached_next_obs = jax.tree.map(move_to_device, timestep.observation)
                    cached_next_dones = move_to_device(next_dones)
                    cached_next_trunc = move_to_device(next_trunc)

                    with RecordTimeTo(timings_dict["compute_action_time"]):
                        rng, key = split_key_fn(rng)
                        pi = actor_apply_fn(params.actor_params, cached_next_obs)
                        value = critic_apply_fn(params.critic_params, cached_next_obs)
                        action = pi.sample(seed=key)
                        log_prob = pi.log_prob(action)

                    with RecordTimeTo(timings_dict["put_action_on_cpu_time"]):
                        action_cpu = np.array(jax.device_put(action, cpu))

                    with RecordTimeTo(timings_dict["env_step_time"]):
                        timestep = envs.step(action_cpu)

                    # Get the next dones and truncation flags
                    next_dones = np.logical_and(
                        np.array(timestep.last()), np.array(timestep.discount == 0.0)
                    )
                    next_trunc = np.logical_and(
                        np.array(timestep.last()), np.array(timestep.discount == 1.0)
                    )

                    # Append data to storage
                    reward = timestep.reward
                    info = timestep.extras
                    traj.append(
                        PPOTransition(
                            cached_next_dones,
                            cached_next_trunc,
                            action,
                            value,
                            reward,
                            log_prob,
                            cached_next_obs,
                            info,
                        )
                    )

                # Send the trajectory to the pipeline
                with RecordTimeTo(timings_dict["rollout_put_time"]):
                    pipeline.put(traj, timestep, timings_dict)

            envs.close()

    return rollout


def get_actor_thread(
    env_builder: EnvFactory,
    actor_device: jax.Device,
    params_source: ParamsSource,
    pipeline: Pipeline,
    apply_fns: Tuple[ActorApply, CriticApply],
    key: chex.PRNGKey,
    config: DictConfig,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
    name: str,
):
    """Get the actor thread."""
    rng = jax.device_put(key, actor_device)

    rollout_fn = get_rollout_fn(
        env_builder,
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
        args=(rng,),
        name=name,
    )

    return actor


def get_learner_update_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> SebulbaLearnerFn[LearnerState, PPOTransition]:
    """Get the sebulba learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: LearnerState, traj_batch: PPOTransition
    ) -> Tuple[LearnerState, Tuple]:

        # CALCULATE ADVANTAGE
        params, opt_states, key, _, last_timestep = learner_state
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
        learner_state = LearnerState(params, opt_states, key, None, last_timestep)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: LearnerState, traj_batch: PPOTransition
    ) -> ExperimentOutput[LearnerState]:
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

        learner_state, (episode_info, loss_info) = _update_step(learner_state, traj_batch)

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def get_learner_rollout_fn(
    learn: SebulbaLearnerFn[LearnerState, PPOTransition],
    config: DictConfig,
    eval_queue: Queue,
    pipeline: Pipeline,
    params_sources: Sequence[ParamsSource],
):
    def learner_rollout(learner_state: LearnerState) -> None:
        for _ in range(config.arch.num_evaluation):
            metrics: List[Tuple[Dict, Dict]] = []
            rollout_times: List[Dict] = []
            learn_timings: Dict[str, List[float]] = defaultdict(list)

            for _ in range(config.system.num_updates_per_eval):
                with RecordTimeTo(learn_timings["rollout_get_time"]):
                    traj_batch, timestep, rollout_time = pipeline.get(block=True)

                learner_state = learner_state._replace(timestep=timestep)
                with RecordTimeTo(learn_timings["learning_time"]):
                    learner_state, episode_metrics, train_metrics = learn(learner_state, traj_batch)

                metrics.append((episode_metrics, train_metrics))
                rollout_times.append(rollout_time)

                unreplicated_params = unreplicate(learner_state.params)

                for source in params_sources:
                    source.update(unreplicated_params)

            # Pass to the evaluator
            episode_metrics, train_metrics = jax.tree.map(lambda *x: np.asarray(x), *metrics)

            rollout_times = jax.tree.map(lambda *x: np.mean(x), *rollout_times)
            timing_dict = rollout_times | learn_timings
            timing_dict = jax.tree.map(np.mean, timing_dict, is_leaf=lambda x: isinstance(x, list))

            eval_queue.put((episode_metrics, train_metrics, learner_state, timing_dict))

    return learner_rollout


def get_learner_thread(
    learn: SebulbaLearnerFn[LearnerState, PPOTransition],
    learner_state: LearnerState,
    config: DictConfig,
    eval_queue: Queue,
    pipeline: Pipeline,
    params_sources: Sequence[ParamsSource],
):

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
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:

    # Get number/dimension of actions.
    env = env_factory(num_envs=1)
    obs_shape = env.unwrapped.single_observation_space.shape
    num_actions = int(env.env.unwrapped.single_action_space.n)
    env.close()
    config.system.action_dim = num_actions

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
    init_x = jnp.ones(obs_shape)
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

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_update_fn(apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

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
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=learner_devices)

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = LearnerState(params, opt_states, None, None, None)

    return learn, apply_fns, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation

    # Get the learner and actor devices
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    assert len(local_devices) == len(
        global_devices
    ), "Local and global devices must be the same for now. We dont support multihost just yet"

    actor_devices = [local_devices[device_id] for device_id in config.arch.actor.device_ids]
    local_learner_devices = [
        local_devices[device_id] for device_id in config.arch.learner.device_ids
    ]
    print(f"{Fore.BLUE}{Style.BRIGHT}[Sebulba] Actors devices: {actor_devices}{Style.RESET_ALL}")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}[Sebulba] Learner devices: {local_learner_devices}{Style.RESET_ALL}"
    )

    config.num_learning_devices = len(local_learner_devices)
    config.num_actor_actor_devices = len(actor_devices)

    # Calculate the number of envs per actor
    num_envs_per_actor_device = config.arch.total_num_envs // len(actor_devices)
    num_envs_per_actor = num_envs_per_actor_device // config.arch.actor.actor_per_device
    config.arch.actor.envs_per_actor = num_envs_per_actor

    # Create the environments for train and eval.
    # env_factory = EnvPoolFactory(
    #     config.arch.seed,
    #     task_id="CartPole-v1",
    #     env_type="dm",
    # )
    env_factory = make_gym_env_factory()

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )
    np_rng = np.random.default_rng(config.arch.seed)

    # Setup learner.
    learn, apply_fns, learner_state = learner_setup(
        env_factory, (key, actor_net_key, critic_net_key), local_learner_devices, config
    )

    # Setup evaluator.
    evaluator, evaluator_envs = get_sebulba_eval_fn(
        env_factory,
        get_distribution_act_fn(config, apply_fns[0]),
        config,
        np_rng,
        absolute_metric=False,
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation

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
    pipeline = Pipeline(config.arch.pipeline_queue_size, local_learner_devices, pipeline_lifetime)
    # Start the pipeline
    pipeline.start()

    params_sources: List[ParamsSource] = []
    actor_threads: List[threading.Thread] = []
    actors_lifetime = ThreadLifetime()
    params_sources_lifetime = ThreadLifetime()
    for actor_device in actor_devices:
        # Create 1 params source per actor device as this will be used to pass the params to the actors
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
    eval_queue: Queue = Queue()
    learner_thread = get_learner_thread(
        learn, learner_state, config, eval_queue, pipeline, params_sources
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
        episode_metrics, train_metrics, learner_state, times_dict = eval_queue.get()

        t = int(steps_per_rollout * (eval_step + 1))
        times_dict["timestep"] = t
        logger.log(times_dict, t, eval_step, LogEvent.MISC)

        episode_metrics, ep_completed = get_final_step_metrics(episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / times_dict["single_rollout_time"]
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)

        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        unreplicated_actor_params = unreplicate(learner_state.params.actor_params)
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = evaluator(unreplicated_actor_params, eval_key, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_state,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(unreplicated_actor_params)
            max_episode_return = episode_return

    evaluator_envs.close()
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Make sure all of the Threads are closed.
    actors_lifetime.stop()
    for actor in actor_threads:
        actor.join()

    pipeline_lifetime.stop()
    pipeline.join()

    params_sources_lifetime.stop()
    for param_source in params_sources:
        param_source.join()

    # Measure absolute metric.
    # if config.arch.absolute_metric:
    #     abs_metric_evaluator, abs_metric_evaluator_envs = get_eval_fn(
    #         environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=True
    #     )
    #     key, eval_key = jax.random.split(key, 2)
    #     eval_metrics = abs_metric_evaluator(best_params, eval_key, {})

    #     t = int(steps_per_rollout * (eval_step + 1))
    #     logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
    #     abs_metric_evaluator_envs.close()

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(config_path="../../../configs", config_name="default_ff_ppo.yaml", version_base="1.2")
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
