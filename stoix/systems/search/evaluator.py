from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig

from stoix.base_types import EvalFn, EvalState, ExperimentOutput
from stoix.systems.search.search_types import RootFnApply, SearchApply
from stoix.utils.jax_utils import unreplicate_batch_dim


def get_search_evaluator_fn(
    env: Environment,
    search_apply_fn: SearchApply,
    root_fn: RootFnApply,
    config: DictConfig,
    log_solve_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Get the evaluator function for search-based agents."""

    def eval_one_episode(params: FrozenDict, init_eval_state: EvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: EvalState) -> EvalState:
            """Step the environment."""
            # PRNG keys.
            key, env_state, last_timestep, step_count, episode_return = eval_state

            # Select action.
            key, root_key, policy_key = jax.random.split(key, num=3)
            obs, model_env_state = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, ...], (last_timestep.observation, env_state)
            )
            root = root_fn(params, obs, model_env_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action

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

    def evaluator_fn(trained_params: FrozenDict, key: chex.PRNGKey) -> ExperimentOutput[EvalState]:
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

        return ExperimentOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
            train_metrics={},
        )

    return evaluator_fn


def search_evaluator_setup(
    eval_env: Environment,
    key_e: chex.PRNGKey,
    search_apply_fn: SearchApply,
    root_fn: Callable,
    params: FrozenDict,
    config: DictConfig,
) -> Tuple[EvalFn, EvalFn, Tuple[FrozenDict, chex.Array]]:
    """Initialise evaluator_fn."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    # Check if solve rate is required for evaluation.
    if hasattr(config.env, "solved_return_threshold"):
        log_solve_rate = True
    else:
        log_solve_rate = False

    eval_apply_fn = search_apply_fn
    evaluator = get_search_evaluator_fn(eval_env, eval_apply_fn, root_fn, config, log_solve_rate)
    absolute_metric_evaluator = get_search_evaluator_fn(
        eval_env,
        eval_apply_fn,
        root_fn,
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
