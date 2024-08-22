import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig

from stoix.base_types import (
    ActFn,
    ActorApply,
    EnvFactory,
    EvalFn,
    EvalState,
    EvaluationOutput,
    RecActFn,
    RecActorApply,
    RNNEvalState,
    RNNObservation,
    SebulbaEvalFn,
)
from stoix.utils.jax_utils import unreplicate_batch_dim


def get_distribution_act_fn(
    config: DictConfig,
    actor_apply: ActorApply,
    rngs: Optional[Dict[str, chex.PRNGKey]] = None,
) -> ActFn:
    """Get the act_fn for a network that returns a distribution."""

    def act_fn(params: FrozenDict, observation: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Get the action from the distribution."""
        if rngs is None:
            pi = actor_apply(params, observation)
        else:
            pi = actor_apply(params, observation, rngs=rngs)
        if config.arch.evaluation_greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=key)
        return action

    return act_fn


def get_rec_distribution_act_fn(config: DictConfig, rec_actor_apply: RecActorApply) -> RecActFn:
    """Get the act_fn for a recurrent network that returns a distribution."""

    def rec_act_fn(
        params: FrozenDict, hstate: chex.Array, observation: RNNObservation, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array]:
        """Get the action from the distribution."""
        hstate, pi = rec_actor_apply(params, hstate, observation)
        if config.arch.evaluation_greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=key)
        return hstate, action

    return rec_act_fn


def get_ff_evaluator_fn(
    env: Environment,
    act_fn: ActFn,
    config: DictConfig,
    log_solve_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An environment instance for evaluation.
        act_fn (callable): The act_fn that returns the action taken by the agent.
        config (dict): Experiment configuration.
        eval_multiplier (int): A scalar that will increase the number of evaluation
            episodes by a fixed factor. The reason for the increase is to enable the
            computation of the `absolute metric` which is a metric computed and the end
            of training by rolling out the policy which obtained the greatest evaluation
            performance during training for 10 times more episodes than were used at a
            single evaluation step.
    """

    def eval_one_episode(params: FrozenDict, init_eval_state: EvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: EvalState) -> EvalState:
            """Step the environment."""
            # PRNG keys.
            key, env_state, last_timestep, step_count, episode_return = eval_state

            # Select action.
            key, policy_key = jax.random.split(key)

            action = act_fn(
                params,
                jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], last_timestep.observation),
                policy_key,
            )

            # Step environment.
            env_state, timestep = env.step(env_state, action.squeeze())

            # Log episode metrics.
            episode_return += timestep.reward
            step_count += 1
            eval_state = EvalState(key, env_state, timestep, step_count, episode_return)
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        # Log solve episode if solve rate is required.
        if log_solve_rate:
            eval_metrics["solve_episode"] = jnp.all(
                final_state.episode_return >= config.env.solved_return_threshold
            ).astype(int)

        return eval_metrics

    def evaluator_fn(trained_params: FrozenDict, key: chex.PRNGKey) -> EvaluationOutput[EvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = (config.arch.num_eval_episodes // n_devices) * eval_multiplier

        key, *env_keys = jax.random.split(key, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(
            jnp.stack(env_keys),
        )
        # Split keys for each core.
        key, *step_keys = jax.random.split(key, eval_batch + 1)
        # Add dimension to pmap over.
        step_keys = jnp.stack(step_keys).reshape(eval_batch, -1)

        eval_state = EvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            step_count=jnp.zeros((eval_batch, 1)),
            episode_return=jnp.zeros_like(timesteps.reward),
        )

        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return EvaluationOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
        )

    return evaluator_fn


def get_rnn_evaluator_fn(
    env: Environment,
    rec_act_fn: RecActFn,
    config: DictConfig,
    scanned_rnn: nn.Module,
    log_solve_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Get the evaluator function for recurrent networks."""

    def eval_one_episode(params: FrozenDict, init_eval_state: RNNEvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: RNNEvalState) -> RNNEvalState:
            """Step the environment."""
            (
                key,
                env_state,
                last_timestep,
                last_done,
                hstate,
                step_count,
                episode_return,
            ) = eval_state

            # PRNG keys.
            key, policy_key = jax.random.split(key)

            # Add a batch dimension and env dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0)[jnp.newaxis, :], last_timestep.observation
            )
            ac_in = (batched_observation, jnp.expand_dims(last_done, axis=0))

            # Run the network.
            hstate, action = rec_act_fn(params, hstate, ac_in, policy_key)

            # Step environment.
            env_state, timestep = env.step(env_state, action[-1].squeeze(0))

            # Log episode metrics.
            episode_return += timestep.reward
            step_count += 1
            eval_state = RNNEvalState(
                key,
                env_state,
                timestep,
                timestep.last().reshape(-1),
                hstate,
                step_count,
                episode_return,
            )
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        # Log solve episode if solve rate is required.
        if log_solve_rate:
            eval_metrics["solve_episode"] = jnp.all(
                final_state.episode_return >= config.env.solved_return_threshold
            ).astype(int)
        return eval_metrics

    def evaluator_fn(
        trained_params: FrozenDict, key: chex.PRNGKey
    ) -> EvaluationOutput[RNNEvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = config.arch.num_eval_episodes // n_devices * eval_multiplier

        key, *env_keys = jax.random.split(key, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(jnp.stack(env_keys))
        # Split keys for each core.
        key, *step_keys = jax.random.split(key, eval_batch + 1)
        # Add dimension to pmap over.
        step_keys = jnp.stack(step_keys).reshape(eval_batch, -1)

        # Initialise hidden state.
        init_hstate = scanned_rnn.initialize_carry(eval_batch)
        init_hstate = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=1), init_hstate)

        # Initialise dones.
        dones = jnp.zeros(
            (eval_batch, 1),
            dtype=bool,
        )

        eval_state = RNNEvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            dones=dones,
            hstate=init_hstate,
            step_count=jnp.zeros((eval_batch, 1)),
            episode_return=jnp.zeros_like(timesteps.reward),
        )

        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return EvaluationOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
        )

    return evaluator_fn


def evaluator_setup(
    eval_env: Environment,
    key_e: chex.PRNGKey,
    eval_act_fn: Union[ActFn, RecActFn],
    params: FrozenDict,
    config: DictConfig,
    use_recurrent_net: bool = False,
    scanned_rnn: Optional[nn.Module] = None,
) -> Tuple[EvalFn, EvalFn, Tuple[FrozenDict, chex.Array]]:
    """Initialise evaluator_fn."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    # Check if solve rate is required for evaluation.
    if hasattr(config.env, "solved_return_threshold"):
        log_solve_rate = True
    else:
        log_solve_rate = False
    # Vmap it over number of agents and create evaluator_fn.
    if use_recurrent_net:
        assert scanned_rnn is not None
        evaluator = get_rnn_evaluator_fn(
            eval_env,
            eval_act_fn,  # type: ignore
            config,
            scanned_rnn,
            log_solve_rate,
        )
        absolute_metric_evaluator = get_rnn_evaluator_fn(
            eval_env,
            eval_act_fn,  # type: ignore
            config,
            scanned_rnn,
            log_solve_rate,
            10,
        )
    else:
        evaluator = get_ff_evaluator_fn(
            eval_env, eval_act_fn, config, log_solve_rate  # type: ignore
        )
        absolute_metric_evaluator = get_ff_evaluator_fn(
            eval_env,
            eval_act_fn,  # type: ignore
            config,
            log_solve_rate,
            10,
        )

    evaluator = jax.pmap(evaluator, axis_name="device")
    absolute_metric_evaluator = jax.pmap(absolute_metric_evaluator, axis_name="device")

    # Broadcast trained params to cores and split keys for each core.
    trained_params = unreplicate_batch_dim(params)
    key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)

    return evaluator, absolute_metric_evaluator, (trained_params, eval_keys)


def get_sebulba_eval_fn(
    env_factory: EnvFactory,
    act_fn: ActFn,
    config: DictConfig,
    np_rng: np.random.Generator,
    device: jax.Device,
    eval_multiplier: float = 1.0,
) -> Tuple[SebulbaEvalFn, Any]:

    eval_episodes = config.arch.num_eval_episodes * eval_multiplier

    # We calculate here the number of parallel envs we can run in parallel.
    # If the total number of episodes is less than the number of parallel envs
    # we will run all episodes in parallel.
    # Otherwise we will run `num_envs` parallel envs and loop enough times
    # so that we do at least `eval_episodes` number of episodes.
    n_parallel_envs = int(min(eval_episodes, config.arch.num_envs))
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)
    envs = env_factory(n_parallel_envs)
    cpu = jax.devices("cpu")[0]
    act_fn = jax.jit(act_fn, device=device)

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        msg = (
            f"Please note that the number of evaluation episodes ({eval_episodes}) is not "
            f"evenly divisible by `num_envs`. As a result, some additional evaluations will be "
            f"conducted. The adjusted number of evaluation episodes is now "
            f"{episode_loops * n_parallel_envs}."
        )
        print(f"{Fore.YELLOW}{Style.BRIGHT}{msg}{Style.RESET_ALL}")

    def eval_fn(params: FrozenDict, key: chex.PRNGKey) -> Dict:
        def _run_episodes(key: chex.PRNGKey) -> Tuple[chex.PRNGKey, Dict]:
            """Simulates `num_envs` episodes."""
            with jax.default_device(device):
                # Reset the environment.
                seeds = np_rng.integers(np.iinfo(np.int32).max, size=n_parallel_envs).tolist()
                timestep = envs.reset(seed=seeds)

                all_metrics = [timestep.extras["metrics"]]
                all_dones = [timestep.last()]
                finished_eps = timestep.last()

                # Loop until all episodes are done.
                while not finished_eps.all():
                    key, act_key = jax.random.split(key)
                    action = act_fn(params, timestep.observation, act_key)
                    action_cpu = np.asarray(jax.device_put(action, cpu))
                    timestep = envs.step(action_cpu)
                    all_metrics.append(timestep.extras["metrics"])
                    all_dones.append(timestep.last())
                    finished_eps = np.logical_or(finished_eps, timestep.last())

                metrics = jax.tree.map(lambda *x: np.stack(x), *all_metrics)
                dones = np.stack(all_dones)

                # find the first instance of done to get the metrics at that timestep, we don't
                # care about subsequent steps because we only the results from the first episode
                done_idx = np.argmax(dones, axis=0)
                metrics = jax.tree_map(lambda m: m[done_idx, np.arange(n_parallel_envs)], metrics)
                del metrics["is_terminal_step"]  # unneeded for logging

                return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes.
        metrics = []
        for _ in range(episode_loops):
            key, metric = _run_episodes(key)
            metrics.append(metric)

        metrics: Dict = jax.tree_map(
            lambda *x: np.array(x).reshape(-1), *metrics
        )  # flatten metrics
        return metrics

    def timed_eval_fn(params: FrozenDict, key: chex.PRNGKey) -> Any:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.perf_counter()
        metrics = eval_fn(params, key)
        end_time = time.perf_counter()

        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        metrics["evaluator_run_time"] = end_time - start_time
        return metrics

    return timed_eval_fn, envs
