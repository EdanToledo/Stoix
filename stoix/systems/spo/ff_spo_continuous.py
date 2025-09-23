import copy
import functools
import time
from typing import Any, Callable, Dict, NamedTuple, Tuple

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
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, TimeStep, WrapperState, get_final_step_metrics
from tensorflow_probability.substrates.jax.distributions import Distribution

from stoix.base_types import (
    Action,
    ActorApply,
    AnakinExperimentOutput,
    CriticApply,
    LearnerFn,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.networks.distributions import AffineTanhTransformedDistribution
from stoix.systems.mpo.continuous_loss import (
    clip_dual_params,
    compute_cross_entropy_loss,
    compute_nonparametric_kl_from_normalized_weights,
    compute_parametric_kl_penalty_and_dual_loss,
    compute_weights_and_temperature_loss,
)
from stoix.systems.mpo.mpo_types import DualParams
from stoix.systems.search.evaluator import search_evaluator_setup
from stoix.systems.search.search_types import EnvironmentStep
from stoix.systems.spo.spo_types import (
    _SPO_FLOAT_EPSILON,
    SPOApply,
    SPOOptStates,
    SPOOutput,
    SPOParams,
    SPORecurrentFn,
    SPORecurrentFnOutput,
    SPORootFnApply,
    SPORootFnOutput,
    SPOTransition,
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

tfd = tfp.distributions


def check_distribution_type(action_distribution: Distribution) -> None:
    """Verify that the policy's action distribution is of required types.

    This function ensures that the action distribution follows the expected structure
    for the SPO algorithm, which requires an Independent distribution wrapping an
    AffineTanhTransformedDistribution.

    Args:
        action_distribution: The distribution to check, typically produced by the policy.

    Raises:
        ValueError: If the distribution is not an Independent distribution or if
            its wrapped distribution is not an AffineTanhTransformedDistribution.
    """
    if not isinstance(action_distribution, tfd.Independent):
        raise ValueError("Action distribution must be a Independent distribution.")

    if not isinstance(action_distribution.distribution, AffineTanhTransformedDistribution):
        raise ValueError("Action distribution must be AffineTanhTransformedDistribution.")


def broadcast_tree(struct: chex.ArrayTree, add_dims: Tuple[int], axis: int = 0) -> chex.ArrayTree:
    """Add dimensions to each array in a tree structure and broadcast values along those dimensions.

    This function takes a PyTree of arrays and adds additional dimensions at the specified axis,
    then broadcasts the original values across the newly added dimensions.

    Args:
        struct: A PyTree of arrays to be broadcasted.
        add_dims: A tuple specifying the dimensions to add at the specified axis.
        axis: The axis position where new dimensions should be inserted. Default is 0.

    Returns:
        A new PyTree with the same structure but with arrays that have been expanded
        and broadcasted along the new dimensions.
    """

    def broadcast_fn(x: chex.Array) -> chex.Array:
        x_shape = x.shape
        prefix = x_shape[:axis]
        suffix = x_shape[axis:]
        new_shape = prefix + add_dims + suffix
        x = jnp.expand_dims(x, axis=axis)
        return jnp.broadcast_to(x, new_shape)

    return jax.tree_util.tree_map(
        broadcast_fn,
        jax.tree_util.tree_map(jnp.asarray, struct),
    )


def apply_exploration_noise(
    rng_key: chex.PRNGKey,
    base_action: chex.Array,
    noise_scale: float,
    noise_min: float,
    noise_max: float,
) -> chex.Array:
    """Applies bounded exploration noise to a continuous action.

    Perturbs an action by adding scaled noise from a truncated normal distribution,
    ensuring the result stays within specified bounds. The noise is proportionally
    mixed with the original action based on the noise scale.

    Args:
        rng_key: Random key for generating noise.
        base_action: Original action to be perturbed.
        noise_scale: Mixing coefficient between original action and noise (0.0 to 1.0).
        noise_min: Minimum value for the generated noise and output action.
        noise_max: Maximum value for the generated noise and output action.

    Returns:
        Perturbed action within specified bounds with same shape as input action.

    Example:
        action = jnp.array([0.5, -0.3])
        noisy_action = apply_exploration_noise(rng_key, action, 0.1, -1.0, 1.0)
    """
    # Generate truncated normal noise matching action shape
    noise_sample = jax.random.truncated_normal(
        rng_key, lower=noise_min, upper=noise_max, shape=base_action.shape
    )

    # Interpolate between original action and noise
    noisy_action = (1.0 - noise_scale) * base_action + noise_scale * noise_sample

    # Ensure result respects bounds
    return jnp.clip(noisy_action, noise_min, noise_max)


def make_root_fn(
    actor_apply_fn: ActorApply,
    critic_apply_fn: CriticApply,
    config: DictConfig,
) -> SPORootFnApply:
    """Create the root function for initializing SPO search.

    This function returns a callable that generates the root objects needed for
    Sequential Monte Carlo (SMC) search. The root function samples initial actions,
    computes their values and log probabilities, and prepares environment states for
    particle-based simulation.

    Args:
        actor_apply_fn: Function for applying the actor network to observations.
        critic_apply_fn: Function for applying the critic network to observations.
        config: Configuration dictionary containing search parameters.

    Returns:
        A function that takes parameters, observation, environment state, and an RNG key,
        and returns a SPORootFnOutput containing initial particle states and actions.
    """

    def root_fn(
        params: SPOParams,
        observation: chex.ArrayTree,
        env_state: chex.ArrayTree,
        rng_key: chex.PRNGKey,
    ) -> SPORootFnOutput:
        """Initialize the root node for Sequential Monte Carlo (SMC) search.

        This function creates the starting point for SMC search by sampling actions from
        the current policy, evaluating their values, and preparing environment states for
        particle-based simulation.

        Args:
            params: Network parameters including actor, critic, and dual parameters.
            observation: Current environment observation.
            env_state: Current environment state.
            rng_key: Random key for stochastic operations.

        Returns:
            SPORootFnOutput containing initial particle states, actions, values and logits.
        """

        sample_key, noise_key = jax.random.split(rng_key, 2)
        # Run the actor and critic network on the current observation
        pi = actor_apply_fn(params.actor_params.online, observation)
        value = critic_apply_fn(params.critic_params.online, observation)

        # Sample an action for every particle that is going to be used in the SMC search
        sampled_actions = pi.sample(seed=sample_key, sample_shape=config.system.num_particles)
        # Swap num samples and batch dimension
        sampled_actions = jnp.swapaxes(sampled_actions, 0, 1)
        # Check shapes for sanity
        batch_size = value.shape[0]
        chex.assert_shape(
            sampled_actions, (batch_size, config.system.num_particles, config.system.action_dim)
        )
        sampled_actions = apply_exploration_noise(
            noise_key,
            sampled_actions,
            config.system.root_exploration_fraction,
            config.system.action_minimum,
            config.system.action_maximum,
        )

        # We then broadcast the environment state so each particle has a copy and we do the same
        # for the critic value. This is all on the axis=1 so the shape is (batch, num particles, *)
        particle_env_states = broadcast_tree(env_state, (config.system.num_particles,), axis=1)
        value = broadcast_tree(value, (config.system.num_particles,), axis=1)
        # For each sampled action, we get the log prob. This is not strictly necessary to have.
        log_probs = jax.vmap(pi.log_prob, in_axes=1, out_axes=1)(sampled_actions)

        # We create the root object
        root_fn_output = SPORootFnOutput(
            particle_logits=log_probs,
            particle_actions=sampled_actions,
            particle_values=value,
            particle_env_states=particle_env_states,
        )

        return root_fn_output

    return root_fn


def make_recurrent_fn(
    environment_step: EnvironmentStep,
    actor_apply_fn: ActorApply,
    critic_apply_fn: CriticApply,
    config: DictConfig,
) -> SPORecurrentFn:
    """Create the recurrent function for advancing particles during SMC search.

    This function returns a callable that steps particles through the environment model
    during the Sequential Monte Carlo (SMC) search. For each particle, it advances
    the environment state, computes rewards and values, and samples next actions.

    Args:
        environment_step: Function for stepping the environment model forward.
        actor_apply_fn: Function for applying the actor network to observations.
        critic_apply_fn: Function for applying the critic network to observations.
        config: Configuration dictionary containing search parameters.

    Returns:
        A function that takes parameters, RNG key, particle actions, and environment states,
        and returns updated states and recurrent function outputs for each particle.
    """

    def check_shapes(
        recurrent_fn_output: SPORecurrentFnOutput, particle_actions: chex.Array
    ) -> None:
        """Verify that recurrent function outputs have the expected shapes.

        This function performs shape assertions to ensure that the outputs from the
        recurrent function match the expected shapes based on the configuration.

        Args:
            recurrent_fn_output: Output from the recurrent function step.
            particle_actions: Actions taken by the particles.

        Raises:
            AssertionError: If any of the shapes don't match expectations.
        """
        chex.assert_shape(recurrent_fn_output.reward, (config.system.num_particles,))
        chex.assert_shape(recurrent_fn_output.discount, (config.system.num_particles,))
        chex.assert_shape(recurrent_fn_output.prior_logits, (config.system.num_particles,))
        chex.assert_shape(recurrent_fn_output.value, (config.system.num_particles,))
        chex.assert_shape(
            recurrent_fn_output.next_sampled_action,
            (config.system.num_particles, particle_actions.shape[1]),
        )

    def recurrent_fn(
        params: SPOParams,
        rng_key: chex.PRNGKey,
        particle_actions: chex.Array,
        env_state: chex.ArrayTree,
    ) -> Tuple[SPORecurrentFnOutput, chex.ArrayTree]:
        """Execute one step of the environment for each particle during SMC search.

        This function advances the environment state for each particle using its selected action,
        computes new values and rewards, and samples new actions for the next state. It is designed
        to be vmapped across the batch dimension during search.

        Args:
            params: Network parameters including actor, critic, and dual parameters.
            rng_key: Random key for stochastic operations.
            particle_actions: Actions for each particle to take in the environment.
            env_state: Current environment state for each particle.

        Returns:
            A tuple containing:
                - SPORecurrentFnOutput with rewards, discounts, value estimates and next actions.
                - Updated environment state after taking the action.
        """

        # Take a step in the environment using the particles action
        # This environment step is vmapped in the learner setup.
        next_env_state, next_timestep = environment_step(env_state, particle_actions)
        # For each particle, run the actor and critic networks
        pi = actor_apply_fn(params.actor_params.online, next_timestep.observation)
        value = critic_apply_fn(params.critic_params.online, next_timestep.observation)
        # Sample a new action for the new state for each particle
        next_sampled_actions = pi.sample(seed=rng_key)
        # Check shape for sanity
        chex.assert_shape(
            next_sampled_actions, (config.system.num_particles, config.system.action_dim)
        )
        # Check to see if the environment truncated
        truncated_step = next_timestep.last() & (next_timestep.discount != 0.0)
        # We set the discount to 0.0 if the environment truncated
        # The reason for this is to have an indication of if a particle terminated one way or
        # another. However, to still utilise the value for bootstrapping, we manually discount
        # the value by the correct discount in the recurrent_fn_output.
        rec_fn_discount = next_timestep.discount * (1 - truncated_step.astype(jnp.float32))
        # For the bootstrap value, we use the real discount and multiply it by the gamma here.
        bootstrap_value = next_timestep.discount * config.system.search_gamma * value
        # Get the log probabilities of the next sampled actions
        next_log_probs = pi.log_prob(next_sampled_actions)

        # Create the recurrent_fn_output
        recurrent_fn_output = SPORecurrentFnOutput(
            reward=next_timestep.reward,
            discount=rec_fn_discount,
            prior_logits=next_log_probs,
            value=bootstrap_value,
            next_sampled_action=next_sampled_actions,
        )
        # Check shapes for sanity
        check_shapes(recurrent_fn_output, particle_actions)

        return recurrent_fn_output, next_env_state

    return recurrent_fn


class Particles(NamedTuple):
    """Container for particle states used in Sequential Monte Carlo (SMC) search.

    This class stores the state of all particles being simulated during the SMC search.
    Each field has the leading dimensions [NumEnvs, NumParticles, ...] for batch processing.

    Attributes:
        state_embedding: Environment states for each particle.
        root_actions: Actions sampled at the root for each particle.
        resample_td_weights: Temporal difference weights used for resampling decisions.
        prior_logits: Log probabilities of each particle's action under the current policy.
        value: Value estimates for each particle's state.
        terminal: Boolean flags indicating if particles have reached terminal states.
        depth: The depth/step count of each particle in its trajectory.
        gae: Generalized Advantage Estimates for each particle.
    """

    state_embedding: chex.ArrayTree
    root_actions: chex.Array
    resample_td_weights: chex.Array
    prior_logits: chex.Array
    value: chex.Array
    terminal: chex.Array
    depth: chex.Array
    gae: chex.Array


class SPO:
    """Sequential Policy Optimization Search.

    This class implements the Sequential Monte Carlo (SMC) search process for the
    Sequential Policy Optimization (SPO) algorithm. It manages the creation,
    advancement, and resampling of particles to explore possible action trajectories
    and improve policy performance.

    The search process involves:
    1. Initializing particles from a root state
    2. Advancing particles through the environment model
    3. Resampling particles based on their performance
    4. Selecting the best actions based on the search results

    Attributes:
        config: Configuration parameters for the SPO algorithm.
        recurrent_fn: Function used to step particles through the environment.
    """

    def __init__(
        self,
        config: DictConfig,
        recurrent_fn: mctx.RecurrentFn,
    ):
        self.config = config
        self.recurrent_fn = recurrent_fn

    def search(
        self,
        params: SPOParams,
        rng_key: chex.PRNGKey,
        root: SPORootFnOutput,
    ) -> SPOOutput:
        """
        Perform a Sequential Monte Carlo (SMC) search to explore possible action trajectories.

        Args:
            params (SPOParams): Parameters containing actor, critic, and dual network parameters.
            rng_key (chex.Array): Random number generator key for stochastic operations.
            root (SPORootFnOutput): Output from the root function, including initial state
                embeddings and sampled actions.

        Returns:
            SPOOutput: The result of the search, including the final set of actions,
            the weights associated with these actions, the mean value over the actions,
            and advantages associated with the "initial" (not final) set of sampled actions to use
            for optimising the dual temperature parameter.
        """

        # Determine the number of parallel environments (batch size)
        batch_size = root.particle_values.shape[0]
        rng_key, rollout_key = jax.random.split(rng_key, num=2)

        rng_keys = jax.random.split(rng_key, batch_size)

        # Execute the SMC rollout to generate particle trajectories
        particles, rollout_metrics, last_resample = self.rollout(
            params,
            root,
            rollout_key,
        )

        def readout_weighted(data: Tuple[Particles, SPORootFnOutput, chex.Array]) -> SPOOutput:
            """Select action from particle set using temperature-scaled weights."""
            particles, root, rng_key = data

            if self.config.system.temperature.adaptive:
                normalised_action_logits = self.get_resample_logits(
                    particles.resample_td_weights,
                    log_temperature=params.dual_params.log_temperature,
                )
            else:
                normalised_action_logits = self.get_resample_logits(
                    particles.resample_td_weights,
                    temperature=self.config.system.temperature.fixed_temperature,
                )

            action_index = jax.random.categorical(rng_key, logits=normalised_action_logits)
            action_weights = jax.nn.softmax(normalised_action_logits, axis=-1)
            action = particles.root_actions[action_index]

            output = SPOOutput(
                action=action,
                sampled_action_weights=action_weights,
                sampled_actions=particles.root_actions,
                value=jnp.mean(root.particle_values, axis=-1),
                sampled_advantages=particles.gae,
                rollout_metrics=rollout_metrics,
            )

            return output

        output: SPOOutput = jax.vmap(readout_weighted, in_axes=(0))(
            (particles, root, rng_keys),
        )

        return output

    def rollout(
        self,
        params: SPOParams,
        root: SPORootFnOutput,
        rng_key: chex.PRNGKey,
    ) -> Tuple[Particles, Dict[str, chex.Array], chex.Array]:
        """
        Execute a Sequential Monte Carlo (SMC) rollout to explore action trajectories.

        This process involves:
            1. Initializing particles from the root state.
            2. Iteratively advancing particles through the environment using the recurrent function.
            3. Applying resampling based on temporal difference (TD) weights at specified intervals.
            4. Accumulating metrics such as values and advantages across particles.

        Args:
            params (SPOParams): Parameters containing actor, critic, and dual network parameters.
            root (SPORootFnOutput): Output from the root function, including initial state
                embeddings and actions.
            rng_key (chex.PRNGKey): Random number generator key for stochastic operations.

        Returns:
            Particles: The final state of all particles after the rollout, including embeddings,
                weights, values, and metrics.

        Note:
            - The number of rollout steps is determined by `config.system.search_depth`.
            - Each particle represents a potential trajectory through states and actions.
        """

        keys = jax.random.split(rng_key, self.config.system.search_depth)

        # Initialize particles from the root state
        initial_particles = self.init_particles(root)
        initial_sampled_actions = initial_particles.root_actions
        carry = (initial_particles, initial_sampled_actions)
        # Scan over depth and record ESS (effective sample size) at each step.
        final_carry, scan_metrics = jax.lax.scan(
            functools.partial(
                self.one_step_rollout,
                params=params,
            ),
            init=carry,
            xs=(jnp.arange(self.config.system.search_depth), keys),
        )
        (particles, _) = final_carry

        # Process the accumulated metrics from the scan
        rollout_metrics = {}
        num_steps = scan_metrics["ess"].shape[0]
        for d in range(1, num_steps + 1):
            rollout_metrics[f"ess_fraction_depth:{d}"] = (
                scan_metrics["ess"][d - 1] / self.config.system.num_particles
            )
            rollout_metrics[f"entropy_depth:{d}"] = scan_metrics["entropy"][d - 1]
            rollout_metrics[f"mean_td_weights_depth:{d}"] = scan_metrics["mean_td_weights"][d - 1]
            rollout_metrics[f"particles_alive_depth:{d}"] = scan_metrics["particles_alive"][d - 1]
            rollout_metrics[f"resample_depth:{d}"] = scan_metrics["resample"][d - 1]

        last_resample = scan_metrics["resample"][-1]

        return particles, rollout_metrics, last_resample  # type: ignore

    def one_step_rollout(
        self,
        particles_and_actions: Tuple[Particles, Action],
        depth_count_and_key: Tuple[chex.Array, chex.PRNGKey],
        params: SPOParams,
    ) -> Tuple[Tuple[Particles, Action], Dict[str, chex.Array]]:
        """
        Execute a single step of the SMC rollout process.

        This involves:
            1. Advancing all particles by one action step using the recurrent function.
            2. Updating temporal difference (TD) weights based on rewards and value estimates.
            3. Conditionally resampling particles based on the resampling period.

        Args:
            particles_and_actions (Tuple[Particles, Action]):
                - particles: Current particle states and their metrics.
                - sampled_actions: Actions sampled for each particle.
            depth_count_and_key (Tuple[chex.Array, chex.PRNGKey]):
                - current_depth: Current depth level in the rollout.
                - key: RNG key for stochastic operations.
            params (SPOParams): Parameters containing actor, critic, and dual network parameters.

        Returns:
            Tuple containing:
                - Updated (particles, sampled_actions) tuple for the next step.
                - None (placeholder for JAX scan compatibility).
        """

        # Unpack the current particles and actions
        particles, sampled_actions = particles_and_actions

        # Ensure the sampled actions have the correct shape
        chex.assert_shape(
            sampled_actions,
            (
                particles.value.shape[0],
                self.config.system.num_particles,
                self.config.system.action_dim,
            ),
        )

        # Unpack the current depth and RNG key
        current_depth, key = depth_count_and_key

        # Split the RNG key for resampling and recurrent steps
        key_resampling, recurrent_step_key = jax.random.split(key)
        batch_recurrent_step_keys = jax.random.split(recurrent_step_key, particles.value.shape[0])

        # Advance the environment by one step for all particles using the recurrent function
        recurrent_output, next_state_embedding = jax.vmap(
            self.recurrent_fn, in_axes=(None, 0, 0, 0), out_axes=(0, 0)
        )(
            params,
            batch_recurrent_step_keys,
            sampled_actions,
            particles.state_embedding,
        )
        next_sampled_actions = recurrent_output.next_sampled_action

        # Update temporal difference (TD) weights based on rewards and value estimates
        updated_td_weights = self.smc_weight_update_fn(
            particles=particles,
            recurrent_output=recurrent_output,
        )
        # Get the terminal mask for the particles for logging
        particles_alive = 1 - particles.terminal.astype(jnp.int32)

        # --- Compute ESS before resampling ---
        if self.config.system.temperature.adaptive:
            ess, entropy = self.calculate_ess_and_entropy(
                updated_td_weights, log_temperature=params.dual_params.log_temperature
            )
        else:
            ess, entropy = self.calculate_ess_and_entropy(
                updated_td_weights, temperature=self.config.system.temperature.fixed_temperature
            )

        root_action = jnp.where(
            current_depth == 0,
            sampled_actions,
            particles.root_actions,
        )

        # Update particle states with new embeddings, weights, actions, and metrics
        updated_particles = self.update_particles(
            next_state_embedding,
            updated_td_weights,
            root_action,
            recurrent_output,
            particles,
        )

        # Compute logits for resampling based on updated TD weights and temperature
        if self.config.system.temperature.adaptive:
            resample_logits = self.get_resample_logits(  # Fix this if we want a fixed temperature
                updated_td_weights,
                log_temperature=params.dual_params.log_temperature,
            )
        else:
            resample_logits = self.get_resample_logits(  # Fix this if we want a fixed temperature
                updated_td_weights,
                temperature=self.config.system.temperature.fixed_temperature,
            )

        # Decide whether to resample based on the configured mode.
        resampling_mode = self.config.system.resampling.mode

        # Calculate metrics to return
        step_metrics = {
            "ess": ess,
            "entropy": entropy,
            "mean_td_weights": updated_td_weights.mean(axis=-1),
            "particles_alive": particles_alive.mean(axis=-1),
        }

        if resampling_mode == "period":
            # Check if (current_depth+1) satisfies the period condition.
            should_resample = ((current_depth + 1) % self.config.system.resampling.period) == 0

            batch_size = updated_particles.root_actions.shape[0]
            step_metrics["resample"] = should_resample.repeat(batch_size)

            # Conditionally resample particles if the resampling period is met
            updated_particles = jax.lax.cond(
                should_resample,
                lambda _: self.resample(updated_particles, key_resampling, resample_logits),
                lambda _: updated_particles,
                None,
            )

            return (updated_particles, next_sampled_actions), step_metrics

        elif resampling_mode == "ess":
            # Resample if the ESS fraction is below the provided threshold.
            # Here, `ess` is a vector so that each batch element can be treated independently.
            condition = ess < (
                self.config.system.resampling.ess_threshold * self.config.system.num_particles
            )

            step_metrics["resample"] = condition

            # Compute resampled particles for all batch elements.
            resampled_particles = self.resample(updated_particles, key_resampling, resample_logits)

            # Element-wise, if condition[i] is True, select the resampled particle; otherwise,
            # keep the original.
            def select_fn(new_field: chex.Array, old_field: chex.Array) -> chex.Array:
                # Broadcast condition to match the shape of each field.
                cond = condition.reshape((condition.shape[0],) + (1,) * (old_field.ndim - 1))
                return jnp.where(cond, new_field, old_field)

            updated_particles = jax.tree_util.tree_map(
                select_fn, resampled_particles, updated_particles
            )
            return (updated_particles, next_sampled_actions), step_metrics

        else:
            raise ValueError(f"Invalid resampling mode: {resampling_mode}")

    def init_particles(self, root: SPORootFnOutput) -> Particles:
        """
        Initialize particles at the start of the search.

        Args:
            root (SPORootFnOutput): The root object containing initial environment
                states and actions.

        Returns:
            Particles: Initialized particles with initial states, actions, and default metrics.
        """

        batch_size = root.particle_values.shape[0]
        particles = Particles(
            state_embedding=root.particle_env_states,
            root_actions=root.particle_actions,
            resample_td_weights=jnp.zeros(shape=(batch_size, self.config.system.num_particles)),
            prior_logits=root.particle_logits,
            value=root.particle_values,
            terminal=jnp.zeros(
                shape=(
                    batch_size,
                    self.config.system.num_particles,
                ),
                dtype=jnp.bool,
            ),
            depth=jnp.zeros(
                shape=(
                    batch_size,
                    self.config.system.num_particles,
                ),
                dtype=jnp.int32,
            ),
            gae=jnp.zeros(shape=(batch_size, self.config.system.num_particles), dtype=jnp.float32),
        )
        # Check shape for sanity
        chex.assert_tree_shape_prefix(
            particles,
            (
                batch_size,
                self.config.system.num_particles,
            ),
        )

        return particles

    def smc_weight_update_fn(
        self,
        particles: Particles,
        recurrent_output: SPORecurrentFnOutput,
    ) -> chex.Array:
        """
        Update temporal difference (TD) weights based on rewards and value estimates.

        Args:
            particles (Particles): Current particle states and metrics.
            recurrent_output (SPORecurrentFnOutput): Output from the recurrent function,
                including rewards and next values.

        Returns:
            chex.Array: Updated TD weights for each particle.
        """
        # Compute the TD error: reward + next value - current value
        # We do not multiply by discount as we do it in the recurrent_fn.
        td_error = recurrent_output.reward + recurrent_output.value - particles.value

        # Apply a terminal mask to ignore updates for terminal states
        terminal_mask = 1 - particles.terminal.astype(jnp.int32)

        # Update TD weights by accumulating the TD error, considering the terminal mask
        # we do not want to add td errors after autoresetting or particles die.
        next_td_weights = td_error * terminal_mask + particles.resample_td_weights

        # Validate the shape of the updated TD weights
        chex.assert_shape(
            next_td_weights,
            (
                particles.value.shape[0],
                self.config.system.num_particles,
            ),
        )

        return next_td_weights

    def get_resample_logits(
        self,
        td_weights: chex.Array,
        log_temperature: chex.Array | None = None,
        temperature: float | None = None,
    ) -> chex.Array:
        """
        Compute resampling logits from temporal difference (TD) weights and temperature.

        Args:
            td_weights (chex.Array): Temporal difference weights for each particle.
            log_temperature (chex.Array): Logarithm of the temperature parameter for scaling.

        Returns:
            chex.Array: Logits used for categorical resampling of particles.
        """

        # Convert log temperature to temperature and ensure numerical stability
        if log_temperature is not None:
            temperature = jax.nn.softplus(log_temperature).squeeze() + _SPO_FLOAT_EPSILON

        # Scale TD weights by temperature to obtain logits
        return td_weights / temperature

    def resample(
        self,
        particles: Particles,
        key: chex.Array,
        resample_logits: chex.Array,
    ) -> Particles:
        """
        Resample particles based on computed logits to focus on promising trajectories.

        Args:
            particles (Particles): Current particle states and metrics.
            key (chex.Array): RNG key for stochastic resampling.
            resample_logits (chex.Array): Logits determining the probability of selecting
                each particle.

        Returns:
            Particles: Resampled particles with reset TD weights and preserved advantages.
        """

        # Split the RNG key for resampling operations
        key, key_resample = jax.random.split(key)

        # Generate separate keys for each batch dimension
        batch_dim_keys = jax.random.split(key_resample, resample_logits.shape[0])
        # Sample indices for resampling using categorical distribution based on logits
        particle_selection_idxs = jax.vmap(jax.random.categorical, in_axes=(0, 0, None, None))(
            batch_dim_keys,
            resample_logits,
            -1,
            (self.config.system.num_particles,),
        )

        def get_particles(
            selection_idxs: chex.Array,
            non_batch_particles: Particles,
        ) -> Particles:
            """
            Select particles based on sampled indices for a single batch.

            Args:
                selection_idxs (chex.Array): Indices of selected particles for resampling.
                non_batch_particles (Particles): Particles from a single batch.

            Returns:
                Particles: Resampled particles for the batch.
            """
            return jax.tree_util.tree_map(  # type: ignore
                lambda x: x[selection_idxs], non_batch_particles
            )

        # Apply resampling across all batches
        particles_resampled = jax.vmap(get_particles, in_axes=(0, 0))(
            particle_selection_idxs,
            particles,
        )

        batch_size = particles.value.shape[0]

        # Reset the TD weights after resampling
        particles_resampled: Particles = particles_resampled._replace(
            resample_td_weights=jnp.zeros(shape=(batch_size, self.config.system.num_particles)),
        )
        # Preserve the Generalized Advantage Estimation (GAE) before resampling.
        # For the temperature loss, to correctly target the KL, we need use the advantages
        # before resampling has occurred. This means that the GAE/Advantage used is only calculated
        # up to the resampling period however, this is still good enough for the adaptive
        # temperature loss as it is still a suitable approximation of the advantage of sampled
        # actions from the policy.
        return particles_resampled._replace(gae=particles.gae)

    def update_particles(
        self,
        embedding: chex.ArrayTree,
        updated_td_weights: chex.Array,
        root_action: chex.Array,
        recurrent_output: SPORecurrentFnOutput,
        particles: Particles,
    ) -> Particles:
        """
        Update particle states with new embeddings, weights, actions, and metrics.

        Args:
            embedding (chex.ArrayTree): New state embeddings from the environment after action
                execution.
            updated_td_weights (chex.Array): Updated temporal difference weights.
            root_action (chex.Array): Actions taken from the root node.
            recurrent_output (SPORecurrentFnOutput): Output from the recurrent function, including
                new values and rewards.
            particles (Particles): Current particle states and metrics.

        Returns:
            Particles: Updated particle states with new embeddings, actions, weights, and metrics.
        """

        return Particles(
            state_embedding=embedding,
            root_actions=root_action,
            resample_td_weights=updated_td_weights,
            prior_logits=recurrent_output.prior_logits,
            value=recurrent_output.value,
            terminal=jnp.logical_or(
                particles.terminal, jnp.where(recurrent_output.discount == 0, True, False)
            ),
            depth=particles.depth + 1,
            gae=self.calculate_gae(
                current_gae=particles.gae,
                value=particles.value,
                next_value=recurrent_output.value,
                reward=recurrent_output.reward,
                discount=recurrent_output.discount,
                depth=particles.depth,
                gamma=self.config.system.search_gamma,
                lambda_=self.config.system.search_gae_lambda,
            ),
        )

    def calculate_gae(
        self,
        current_gae: chex.Array,
        value: chex.Array,
        next_value: chex.Array,
        reward: chex.Array,
        discount: chex.Array,
        depth: chex.Array,
        gamma: float,
        lambda_: float,
    ) -> chex.Array:
        """
        Calculate the Generalized Advantage Estimation (GAE) for each particle.
        This is an iterative calculation going forward in time, not backwards
        as usually done. This is calculated so we can optimise the temperature loss.

        Args:
            current_gae (chex.Array): Current GAE estimates.
            value (chex.Array): Current value estimates.
            next_value (chex.Array): Next value estimates from the recurrent function.
            reward (chex.Array): Rewards received after taking actions.
            discount (chex.Array): Discount factors (typically gamma) applied to future rewards.
            depth (chex.Array): Current depth in the rollout.
            gamma (float): Discount factor for future rewards.
            lambda_ (float): GAE lambda parameter for bias-variance trade-off.

        Returns:
            chex.Array: Updated GAE estimates for each particle.
        """
        # Compute the TD error (delta)
        delta = reward + next_value - value

        # Update GAE using the TD error and decay factors
        updated_gae_estimate = delta * (gamma * lambda_ * discount) ** (depth) + current_gae

        return updated_gae_estimate

    def calculate_ess_and_entropy(
        self,
        td_weights: chex.Array,
        log_temperature: chex.Array | None = None,
        temperature: float | None = None,
    ) -> chex.Array:
        """
        Calculate the Effective Sample Size (ESS) for a given set of TD weights and temperature.
        The ESS is defined as 1 / sum(w_i^2), where w_i are normalized weights computed using
        softmax.

        Args:
            td_weights (chex.Array): Temporal difference weights for each particle.
            log_temperature (chex.Array): Optional log temperature for adaptive scaling.
            temperature (float): Optional fixed temperature if not using adaptive scaling.

        Returns:
            chex.Array: Effective Sample Size per batch element.
        """

        # Compute the scaled logits from the TD weights.
        logits = self.get_resample_logits(td_weights, log_temperature, temperature)

        # Normalize the logits to obtain probabilities.
        weights = jax.nn.softmax(logits, axis=-1)

        # ESS is the inverse of the sum of squared normalized weights.
        ess = 1.0 / jnp.sum(weights**2, axis=-1)

        # Compute the entropy of the weights.
        entropy = -jnp.sum(weights * jnp.log(weights + jnp.finfo(weights.dtype).tiny), axis=-1)

        return ess, entropy


def get_warmup_fn(
    env: Environment,
    params: SPOParams,
    apply_fns: Tuple[ActorApply, CriticApply, SPORootFnApply, SPOApply],
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    """Create a function for warming up the replay buffer before training.

    This function generates trajectories for initial buffer population by stepping
    through the environment using the current policy.

    Args:
        env: Environment to interact with.
        params: Network parameters including actor, critic, and dual parameters.
        apply_fns: Tuple of network apply functions (actor, critic, root_fn, search_apply_fn).
        buffer_add_fn: Function for adding transitions to the replay buffer.
        config: Configuration dictionary containing algorithm parameters.

    Returns:
        A function that steps through the environment and adds transitions to the replay buffer.
    """

    _, _, root_fn, search_apply_fn = apply_fns

    def warmup(
        env_states: WrapperState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[WrapperState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[WrapperState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[WrapperState, TimeStep, chex.PRNGKey], SPOTransition]:
            """Execute a single environment step during policy rollout."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, root_key, policy_key = jax.random.split(key, num=3)
            root = root_fn(params, last_timestep.observation, env_state.unwrapped_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action
            search_policy = search_output.sampled_action_weights
            search_value = search_output.value
            sampled_actions = search_output.sampled_actions

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated_step = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]
            # Get the timestep's (potentially final) observation for bootstrapping.
            # This is done to support truncated episodes.
            bootstrap_obs = timestep.extras["next_obs"]

            transition = SPOTransition(
                done,
                truncated_step,
                action,
                sampled_actions,
                search_policy,
                timestep.reward,
                search_value,
                last_timestep.observation,
                bootstrap_obs,
                info,
                search_output.sampled_advantages,
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
    apply_fns: Tuple[ActorApply, CriticApply, SPORootFnApply, SPOApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[OffPolicyLearnerState]:
    """Create the learner function for training the SPO agent.

    This function builds a learner for the Sequential Policy Optimization (SPO) algorithm.
    The learner function handles the entire training loop, including:
    1. Collecting experience by interacting with the environment
    2. Storing transitions in the replay buffer
    3. Sampling experience batches for training
    4. Computing actor, critic, and dual losses
    5. Updating network parameters

    Args:
        env: Environment to interact with during training.
        apply_fns: Tuple of network apply functions (actor, critic, root_fn, search_apply_fn).
        update_fns: Tuple of optimizer update functions for actor, critic, and dual networks.
        buffer_fns: Tuple of buffer functions (add_fn, sample_fn) for replay buffer operations.
        config: Configuration dictionary containing algorithm parameters.

    Returns:
        A learner function that takes a learner state and returns updated state and metrics.
    """

    # Unpack apply functions for actor, critic, SPO root, and SPO search.
    actor_apply_fn, critic_apply_fn, root_fn, search_apply_fn = apply_fns

    # Unpack update functions for actor, critic, and dual networks.
    actor_update_fn, critic_update_fn, dual_update_fn = update_fns

    # Unpack buffer functions for adding and sampling trajectories.
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        """Execute a single update step of the SPO training loop.

        This function performs a complete update cycle, including:
        1. Collecting experience by rolling out the current policy in the environment
        2. Adding new transitions to the replay buffer
        3. Sampling experiences from the buffer
        4. Computing and applying updates to network parameters

        Args:
            learner_state: Current state of the learner including network parameters,
                optimizer states, buffer state, and environment state.
            _: Dummy argument for compatibility with jax.lax.scan.

        Returns:
            A tuple containing:
                - Updated learner state after the update step.
                - Metrics from the environment interactions and optimization.
        """

        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Tuple[SPOTransition, Dict[str, chex.Array]]]:
            """Execute a single environment step during policy rollout."""
            params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, root_key, policy_key = jax.random.split(key, num=3)
            root = root_fn(params, last_timestep.observation, env_state.unwrapped_state, root_key)
            search_output = search_apply_fn(params, policy_key, root)
            action = search_output.action
            search_policy = search_output.sampled_action_weights
            search_value = search_output.value
            sampled_actions = search_output.sampled_actions

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated_step = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]
            # Get the timestep's (potentially final) observation for bootstrapping.
            # This is done to support truncated episodes.
            bootstrap_obs = timestep.extras["next_obs"]

            transition = SPOTransition(
                done,
                truncated_step,
                action,
                sampled_actions,
                search_policy,
                timestep.reward,
                search_value,
                last_timestep.observation,
                bootstrap_obs,
                info,
                search_output.sampled_advantages,
            )
            learner_state = OffPolicyLearnerState(
                params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, (transition, search_output.rollout_metrics)

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, (traj_batch, search_metrics) = jax.lax.scan(
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
                online_actor_params: FrozenDict,
                dual_params: DualParams,
                target_actor_params: FrozenDict,
                sequence: SPOTransition,
            ) -> Tuple:
                """Calculate the actor loss."""

                # Merge leading dimensions for batch processing.
                sequence = jax.tree.map(lambda x: merge_leading_dims(x, 2), sequence)
                batch_size = sequence.reward.shape[0]

                # Prepare advantages for temperature loss computation.
                adv_for_temp_loss = sequence.sampled_advantages
                adv_for_temp_loss = jnp.swapaxes(adv_for_temp_loss, 0, 1)
                chex.assert_shape(adv_for_temp_loss, (config.system.num_particles, batch_size))

                # Prepare sampled actions.
                sampled_actions = sequence.sampled_actions
                sampled_actions = jnp.swapaxes(sampled_actions, 0, 1)
                chex.assert_shape(
                    sampled_actions,
                    (config.system.num_particles, batch_size, config.system.action_dim),
                )

                # Prepare normalized SMC weights.
                norm_smc_weights = sequence.sampled_actions_weights
                norm_smc_weights = jnp.swapaxes(norm_smc_weights, 0, 1)
                chex.assert_shape(
                    norm_smc_weights,
                    (config.system.num_particles, batch_size),
                )

                # Compute action distributions for online and lagging target parameters.
                online_action_distribution = actor_apply_fn(online_actor_params, sequence.obs)
                target_action_distribution = actor_apply_fn(target_actor_params, sequence.obs)

                # Ensure the distributions are of expected types.
                check_distribution_type(online_action_distribution)
                check_distribution_type(target_action_distribution)

                # Compute temperature and scaling parameters with numerical stability.
                alpha_mean = (
                    jax.nn.softplus(dual_params.log_alpha_mean).squeeze() + _SPO_FLOAT_EPSILON
                )
                alpha_stddev = (
                    jax.nn.softplus(dual_params.log_alpha_stddev).squeeze() + _SPO_FLOAT_EPSILON
                )
                temperature = (
                    jax.nn.softplus(dual_params.log_temperature).squeeze() + _SPO_FLOAT_EPSILON
                )

                # Extract mean and standard deviation from action distributions.
                online_mean = online_action_distribution.distribution.distribution.mean()
                online_scale = online_action_distribution.distribution.distribution.stddev()
                target_mean = target_action_distribution.distribution.distribution.mean()
                target_scale = target_action_distribution.distribution.distribution.stddev()

                # Define batch and action dimensions.
                batch_size = online_mean.shape[0]
                action_dim = online_mean.shape[-1]

                # Compute normalized policy advantages weights (used for metrics)
                # and temperature loss.
                (
                    normalized_policy_adv_weights,
                    loss_temperature,
                ) = compute_weights_and_temperature_loss(
                    adv_for_temp_loss, config.system.epsilon, temperature
                )

                # Compute non-parametric KL divergence from normalized advantages weights.
                # This is used to check if we are achieving the targeted KL.
                kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
                    normalized_policy_adv_weights
                )

                # Define fixed distributions for cross-entropy loss computations.
                fixed_stddev_dist = tfd.Independent(
                    AffineTanhTransformedDistribution(
                        tfd.Normal(loc=online_mean, scale=target_scale),
                        config.system.action_minimum,
                        config.system.action_maximum,
                    ),
                    reinterpreted_batch_ndims=1,
                )
                fixed_mean_dist = tfd.Independent(
                    AffineTanhTransformedDistribution(
                        tfd.Normal(loc=target_mean, scale=online_scale),
                        config.system.action_minimum,
                        config.system.action_maximum,
                    ),
                    reinterpreted_batch_ndims=1,
                )

                # Compute cross-entropy losses for policy mean and standard deviation.
                loss_policy_mean = compute_cross_entropy_loss(
                    sampled_actions, norm_smc_weights, fixed_stddev_dist
                )
                loss_policy_stddev = compute_cross_entropy_loss(
                    sampled_actions, norm_smc_weights, fixed_mean_dist
                )

                if config.system.per_dim_constraining:
                    # Compute KL divergence per action dimension.
                    kl_mean = target_action_distribution.distribution.kl_divergence(
                        fixed_stddev_dist.distribution
                    )
                    kl_stddev = target_action_distribution.distribution.kl_divergence(
                        fixed_mean_dist.distribution
                    )
                    chex.assert_shape(kl_mean, (batch_size, action_dim))
                    chex.assert_shape(kl_stddev, (batch_size, action_dim))
                else:
                    # Compute overall KL divergence without per-dimension constraints.
                    kl_mean = target_action_distribution.kl_divergence(fixed_stddev_dist)
                    kl_stddev = target_action_distribution.kl_divergence(fixed_mean_dist)
                    chex.assert_shape(kl_mean, (batch_size,))
                    chex.assert_shape(kl_stddev, (batch_size,))

                # Compute KL penalties and dual losses for mean and standard deviation.
                loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
                    kl_mean, alpha_mean, config.system.epsilon_mean
                )
                loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
                    kl_stddev, alpha_stddev, config.system.epsilon_stddev
                )

                # Aggregate KL penalties and dual losses.
                loss_kl_penalty = loss_kl_mean + loss_kl_stddev
                loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
                loss_policy = loss_policy_mean + loss_policy_stddev

                # Total loss combines policy, dual, and KL penalties.
                loss = loss_policy + loss_dual + loss_kl_penalty

                loss_info = {
                    "loss_temperature": loss_temperature,
                    "loss_alpha_mean": loss_alpha_mean,
                    "loss_alpha_stddev": loss_alpha_stddev,
                    "loss_policy_mean": loss_policy_mean,
                    "loss_policy_stddev": loss_policy_stddev,
                    "loss_kl_mean": loss_kl_mean,
                    "loss_kl_stddev": loss_kl_stddev,
                    "adaptive_temperature": temperature,
                    "alpha_mean": alpha_mean,
                    "alpha_stddev": alpha_stddev,
                    "kl_nonparametric": kl_nonparametric,
                    "kl_nonparametric_relative": kl_nonparametric / config.system.epsilon,
                    "kl_mean": kl_mean,
                    "kl_stddev": kl_stddev,
                }

                return loss, loss_info

            def _critic_loss_fn(
                online_critic_params: FrozenDict,
                target_critic_params: FrozenDict,
                sequence: SPOTransition,
            ) -> Tuple:
                """Calculation of the critic loss."""

                # Predict current and target values using respective critic networks.
                pred_values = critic_apply_fn(online_critic_params, sequence.obs)

                target_v_tm1 = critic_apply_fn(target_critic_params, sequence.obs)
                target_v_t = critic_apply_fn(target_critic_params, sequence.bootstrap_obs)

                # Compute targets using Generalized Advantage Estimation (GAE).
                _, targets = batch_truncated_generalized_advantage_estimation(
                    sequence.reward,
                    (1 - sequence.done) * config.system.gamma,
                    config.system.gae_lambda,
                    v_tm1=target_v_tm1,
                    v_t=target_v_t,
                    truncation_t=sequence.truncated,
                )

                # Calculate L2 loss between predicted values and targets.
                value_loss = rlax.l2_loss(pred_values, targets).mean()

                # Scale the value loss by a coefficient.
                critic_total_loss = config.system.vf_coef * value_loss

                loss_info = {
                    "value_loss": value_loss,
                    "value_pred_std": pred_values.std(),
                    "value_pred_mean": pred_values.mean(),
                }

                return critic_total_loss, loss_info

            params, opt_states, buffer_state, key = update_state

            key, sample_key = jax.random.split(key, num=2)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: SPOTransition = sequence_sample.experience

            # CALCULATE ACTOR LOSS
            actor_grad_fn = jax.grad(_actor_loss_fn, argnums=(0, 1), has_aux=True)
            actor_dual_grads, actor_dual_loss_info = actor_grad_fn(
                params.actor_params.online,
                params.dual_params,
                params.actor_params.target,
                sequence,
            )

            # CALCULATE CRITIC LOSS
            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(
                params.critic_params.online, params.critic_params.target, sequence
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
            # Separate the gradients
            actor_grads, dual_grads = actor_dual_grads

            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            # pmean over devices.
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )

            # UPDATE OPTIMISER STATE AND ACTOR
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state, params.actor_params.online
            )
            new_online_actor_params = optax.apply_updates(params.actor_params.online, actor_updates)

            # UPDATE OPTIMISER STATE AND CRITIC
            critic_updates, critic_new_opt_state = critic_update_fn(
                critic_grads,
                opt_states.critic_opt_state,
                params.critic_params.online,
            )
            new_online_critic_params = optax.apply_updates(
                params.critic_params.online,
                critic_updates,
            )

            # Update dual network parameters using the optimizer.
            dual_updates, dual_new_opt_state = dual_update_fn(
                dual_grads,
                opt_states.dual_opt_state,
                params.dual_params,
            )
            # Apply updates to dual parameters and enforce constraints.
            dual_new_params = optax.apply_updates(params.dual_params, dual_updates)
            dual_new_params = clip_dual_params(dual_new_params)

            # Incrementally update target parameters towards online parameters.
            new_target_actor_params = optax.incremental_update(
                new_online_actor_params, params.actor_params.target, config.system.tau
            )
            new_target_critic_params = optax.incremental_update(
                new_online_critic_params, params.critic_params.target, config.system.tau
            )

            # PACKING NEW PARAMS AND OPTIMISER STATE
            new_params = SPOParams(
                OnlineAndTarget(new_online_actor_params, new_target_actor_params),
                OnlineAndTarget(new_online_critic_params, new_target_critic_params),
                dual_new_params,
            )
            new_opt_state = SPOOptStates(
                actor_new_opt_state, critic_new_opt_state, dual_new_opt_state
            )

            # PACKING LOSS INFO
            loss_info = {
                **actor_dual_loss_info,
                **critic_loss_info,
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
        loss_info.update(search_metrics)
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:
        """Execute the SPO training loop for a series of update steps.

        This is the main learner function exposed to the training system. It handles
        batching and vectorization of update steps across devices, and wraps results
        in the expected experiment output format.

        Args:
            learner_state: Current state of the learner including network parameters,
                optimizer states, buffer state, and environment state.

        Returns:
            An AnakinExperimentOutput containing the updated learner state, episode metrics,
            and training metrics collected during the update steps.
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


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig, model_env: Environment
) -> Tuple[LearnerFn[OffPolicyLearnerState], SPORootFnApply, SPOApply, OffPolicyLearnerState]:
    """Initialize all components needed for SPO training.

    This function handles the setup of networks, optimizers, environments, and initial states
    required for SPO training. It constructs the actor, critic, and dual networks,
    initializes parameters and optimizer states, creates the environment model for search,
    and prepares the replay buffer.

    Args:
        env: Training environment to interact with.
        keys: PRNG keys for initialization.
        config: Configuration dictionary containing algorithm parameters.
        model_env: Environment model used for SMC search.

    Returns:
        A tuple containing:
            - The learner function for training.
            - The root function for initializing SMC search.
            - The search apply function for executing SMC search.
            - Initial learner state containing parameters, optimizer states, and environment state.
    """
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    action_dim = int(env.action_space().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_space().minimum)
    config.system.action_maximum = float(env.action_space().maximum)

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
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Initialise Dual params and optimiser state.
    if config.system.per_dim_constraining:
        dual_variable_shape = [action_dim]
    else:
        dual_variable_shape = [1]

    log_temperature = jnp.full([1], config.system.init_log_temperature, dtype=jnp.float32)

    log_alpha_mean = jnp.full(
        dual_variable_shape, config.system.init_log_alpha_mean, dtype=jnp.float32
    )

    log_alpha_stddev = jnp.full(
        dual_variable_shape, config.system.init_log_alpha_stddev, dtype=jnp.float32
    )

    dual_params = DualParams(
        log_temperature=log_temperature,
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev,
    )

    dual_lr = make_learning_rate(config.system.dual_lr, config, config.system.epochs)
    dual_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(dual_lr, eps=1e-5),
    )
    dual_opt_state = dual_optim.init(dual_params)

    # Pack params.
    params = SPOParams(
        OnlineAndTarget(actor_params, actor_params),
        OnlineAndTarget(critic_params, critic_params),
        dual_params,
    )

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    root_fn = make_root_fn(actor_network_apply_fn, critic_network_apply_fn, config)
    environment_model_step = jax.vmap(model_env.step)
    model_recurrent_fn = make_recurrent_fn(
        environment_model_step, actor_network_apply_fn, critic_network_apply_fn, config
    )
    search_method = SPO(config, recurrent_fn=model_recurrent_fn)
    search_apply_fn = search_method.search

    # Pack apply and update functions.
    apply_fns = (
        actor_network_apply_fn,
        critic_network_apply_fn,
        root_fn,
        search_apply_fn,
    )
    update_fns = (actor_optim.update, critic_optim.update, dual_optim.update)

    dummy_info = {
        "episode_return": 0.0,
        "episode_length": 0,
        "is_terminal_step": False,
    }

    # Create replay buffer
    dummy_transition = SPOTransition(
        done=jnp.array(False),
        truncated=jnp.array(False),
        action=jnp.zeros(action_dim, dtype=jnp.float32),
        sampled_actions=jnp.zeros((config.system.num_particles, action_dim), dtype=jnp.float32),
        sampled_actions_weights=jnp.ones((config.system.num_particles,)),
        reward=jnp.array(0.0),
        search_value=jnp.array(0.0),
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        bootstrap_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info=dummy_info,
        sampled_advantages=jnp.zeros((config.system.num_particles,)),
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
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))
    opt_states = SPOOptStates(actor_opt_state, critic_opt_state, dual_opt_state)
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
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (config.system.epochs)
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
    config_path="../../configs/default/anakin",
    config_name="default_ff_spo_continuous.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}SPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
